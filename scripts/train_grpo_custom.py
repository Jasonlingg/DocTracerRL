"""Custom GRPO training loop for document exploration.

Trains Qwen2.5-7B LoRA via GRPO using our DocumentExplorationEnv directly.
No verifiers dependency — rollouts go through the gym env, gradients via HF transformers.

Key design choices (informed by DAPO/DR-GRPO/Search-R1 literature):
- batch_size=4 distinct questions per gradient step, group_size=8 rollouts each:
  literature batches 64-512 questions per step to avoid "ping-pong" gradient
  variance from single-question updates; 4 is a cost-feasible middle ground.
  Reduce --steps when raising batch_size to keep total rollouts (cost) similar.
- beta=0.001 KL penalty against frozen base weights (adapter disabled) — prevents
  late-stage format collapse documented when training from an Instruct model with beta=0.
- Fixed token normalization: avoids length bias from per-episode normalization
- Per-token PPO clipping: avoids exponential variance collapse of turn-summed ratios
- Fresh forward pass for rollout action log-probs to eliminate logits warper distortion
- A question's group naturally gets zero advantage-driven gradient when all its
  rewards are identical — no explicit skip needed, and skipping would also skip
  its KL term, which should still regularize the policy.

Usage:
  python scripts/train_grpo_custom.py \
    --sft-checkpoint checkpoints/sft_qwen_7b/final \
    --out checkpoints/grpo_qwen_7b \
    --steps 75 --batch-size 4
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from loguru import logger
from rich.console import Console
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import typer

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.policies.qwen_common import SYSTEM_PROMPT

if TYPE_CHECKING:
    from peft import PeftModel

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Fixed normalization constant — avoids length bias (DR-GRPO style)
NORM_TOKENS = 256
# Max context tokens fed into the gradient pass — truncate from left to bound
# backward memory. At 7B + bfloat16 activations, a 25k-token context needs
# ~14GB for backward; truncating to 2048 keeps peak usage under ~2GB.
MAX_CTX_TOKENS = 2048


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_model(checkpoint: str, trainable: bool, load_in_4bit: bool = True) -> Any:
    from peft import PeftModel, prepare_model_for_kbit_training

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=_bnb_config() if load_in_4bit else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if trainable:
        if load_in_4bit:
            base = prepare_model_for_kbit_training(base)
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=trainable)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _prepare_ctx(ctx_ids: torch.Tensor) -> torch.Tensor:
    """Truncate context from the left to bound backward memory consistently."""
    if ctx_ids.shape[0] > MAX_CTX_TOKENS:
        return ctx_ids[-MAX_CTX_TOKENS:]
    return ctx_ids


@torch.no_grad()
def _compute_token_log_probs(
    model: Any, ctx_ids: torch.Tensor, action_ids: torch.Tensor
) -> torch.Tensor:
    """Compute exact per-token log-probs for action tokens given context.

    Uses a clean forward pass rather than generate()'s scores, avoiding any
    distortion from temperature, top-k, top-p, or other logits warpers.
    """
    if action_ids.numel() == 0:
        return torch.empty(0)

    device = next(model.parameters()).device
    ctx = _prepare_ctx(ctx_ids).to(device)
    act = action_ids.to(device)
    ctx_len = ctx.shape[0]
    act_len = act.shape[0]

    full_ids = torch.cat([ctx, act]).unsqueeze(0)
    logits = model(full_ids).logits[0]
    act_logits = logits[ctx_len - 1 : ctx_len - 1 + act_len]
    act_log_probs = F.log_softmax(act_logits, dim=-1)
    token_lp = act_log_probs.gather(1, act.unsqueeze(1)).squeeze(1)
    return token_lp.detach().cpu()


def _collect_rollout(
    model: Any,
    tokenizer: AutoTokenizer,
    env: DocumentExplorationEnv,
    questions: list[dict],
    q_idx: int,
    max_steps: int = 10,
    temperature: float = 0.8,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], float]:
    """Run one episode. Returns (step_data, reward).

    step_data: list of (context_ids, action_ids, old_token_log_probs) per assistant turn,
    where old_token_log_probs is captured via clean forward pass for the PPO-clip ratio.
    """
    q = questions[q_idx]
    env.reset(question_idx=q_idx)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {q['question']}"},
    ]

    step_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_reward = 0.0

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            ctx_ids = enc.input_ids

            gen_out = model.generate(
                ctx_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            action_ids = gen_out.sequences[0][ctx_ids.shape[1]:].cpu()
            if action_ids.numel() > 0:
                old_token_lp = _compute_token_log_probs(model, ctx_ids[0], action_ids)
            else:
                old_token_lp = torch.empty(0)
            step_data.append((ctx_ids[0].cpu(), action_ids, old_token_lp))

            action = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": action})

            obs, step_reward, done, _ = env.step(action)
            total_reward += step_reward  # accumulates: hit bonuses + final SUBMIT reward
            if done:
                break
            messages.append({"role": "user", "content": obs})

    model.train()
    return step_data, total_reward


# Small KL penalty against the frozen base weights (adapter disabled = reference
# policy). Without this, GRPO with beta=0 is documented to risk "late-stage format
# collapse" — the policy drifts far enough from the SFT reference to start
# generating malformed output in later training. Matches Search-R1/GlobalRAG's value.
KL_BETA = 0.001


def _step_log_prob_and_kl(
    model: Any, ctx_ids: torch.Tensor, action_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Per-token log prob (with grad) and per-token KL-vs-reference sum for one (context, action) pair.

    Reference log-probs come from the same model with the LoRA adapter disabled
    (frozen base weights) — avoids loading a second full model copy. Uses the k3
    KL estimator (always >= 0, low variance) from the GRPO/DeepSeekMath formulation.

    Deletes the full logits tensor immediately after slicing — Qwen2.5 vocab is
    152k so keeping it in the graph wastes ~1GB per forward pass.
    """
    if action_ids.numel() == 0:
        device = next(model.parameters()).device
        zero = torch.empty(0, device=device, requires_grad=True)
        return zero, torch.empty(0, device=device), 0

    device = next(model.parameters()).device
    ctx = _prepare_ctx(ctx_ids).to(device)
    act = action_ids.to(device)

    ctx_len = ctx.shape[0]
    act_len = act.shape[0]

    full_ids = torch.cat([ctx, act]).unsqueeze(0)
    logits = model(full_ids).logits[0]
    act_logits = logits[ctx_len - 1 : ctx_len - 1 + act_len].clone()
    del logits
    torch.cuda.empty_cache()

    act_log_probs = F.log_softmax(act_logits, dim=-1)
    token_lp = act_log_probs.gather(1, act.unsqueeze(1)).squeeze(1)

    with torch.no_grad(), model.disable_adapter():
        ref_logits = model(full_ids).logits[0]
        ref_act_logits = ref_logits[ctx_len - 1 : ctx_len - 1 + act_len]
        ref_log_probs = F.log_softmax(ref_act_logits, dim=-1)
        ref_token_lp = ref_log_probs.gather(1, act.unsqueeze(1)).squeeze(1)
        del ref_logits
        torch.cuda.empty_cache()

    # k3 estimator: exp(ref - policy) - (ref - policy) - 1 — always >= 0, low variance.
    # token_lp keeps its grad here (not detached) so KL backprops into the policy.
    log_ratio = ref_token_lp - token_lp
    kl_per_token = torch.exp(log_ratio) - log_ratio - 1

    return token_lp, kl_per_token, act_len


# PPO-clip range and epoch count — reusing each batch of (expensive-to-generate)
# rollouts for multiple gradient updates instead of one.
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 3


def _grpo_update(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch_step_data: list[list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]],
    batch_rewards: list[list[float]],
    step_num: int = 0,
) -> tuple[float, float, int]:
    """GRPO gradient accumulation across a BATCH of questions, PPO_EPOCHS times.

    Per-token PPO clipping formulation:
    ratio_t = exp(token_lp_t - old_token_lp_t)
    loss = - sum_t min(ratio_t * adv, clip(ratio_t, 1 - eps, 1 + eps) * adv)
    """
    # Dropout MUST be off here. `_collect_rollout` captures old_token_lp under
    # model.eval() and then leaves the model in train() mode on the way out. If the
    # update ran in train mode, LoRA dropout (p=0.05) would perturb this forward pass
    # relative to the one that produced old_token_lp, so ratio != 1 even at step 0
    # with identical weights — silently breaking PPO's importance-sampling assumption
    # that "old" and "new" are comparable. eval() disables dropout WITHOUT disabling
    # gradients (only torch.no_grad() does that), so backward still works.
    model.eval()

    n_uniform = 0
    batch_n = len(batch_rewards)

    batch_advantages: list[list[float]] = []
    for group_rewards in batch_rewards:
        if len({round(r, 4) for r in group_rewards}) == 1:
            n_uniform += 1
        rewards = torch.tensor(group_rewards, dtype=torch.float32)
        batch_advantages.append(
            ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()
        )

    total_loss = 0.0
    total_kl = 0.0
    total_kl_tokens = 0

    for _epoch in range(PPO_EPOCHS):
        optimizer.zero_grad()
        total_loss = 0.0
        total_kl = 0.0
        total_kl_tokens = 0

        for group_step_data, advantages in zip(batch_step_data, batch_advantages):
            n_valid = sum(1 for sd in group_step_data if sd)
            if n_valid == 0:
                continue

            for step_data, adv in zip(group_step_data, advantages):
                if not step_data:
                    continue
                for ctx_ids, action_ids, old_token_lp in step_data:
                    token_lp, kl_per_token, n_tok = _step_log_prob_and_kl(model, ctx_ids, action_ids)
                    if n_tok == 0:
                        continue

                    old_token_lp = old_token_lp.to(token_lp.device)
                    # Per-token importance ratio:
                    ratio = torch.exp(token_lp - old_token_lp)

                    # Diagnostic assertion: at epoch 0 of step 0 before any optimizer update,
                    # forward pass on current weights must match generation forward pass exactly.
                    if _epoch == 0 and step_num == 0:
                        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), (
                            f"Step 0 Epoch 0 ratio mismatch! Mean ratio: {ratio.mean().item():.4f}, "
                            f"Min: {ratio.min().item():.4f}, Max: {ratio.max().item():.4f}"
                        )

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv
                    # Standard per-token PPO clipping surrogate loss:
                    policy_loss = -torch.min(surr1, surr2).sum()
                    kl_sum = kl_per_token.sum()

                    piece = (policy_loss + KL_BETA * kl_sum) / NORM_TOKENS / n_valid / batch_n
                    piece.backward()
                    total_loss += piece.item()
                    total_kl += kl_sum.item()
                    total_kl_tokens += n_tok
                    del token_lp, kl_per_token, piece, ratio, surr1, surr2, policy_loss
                    torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

    mean_kl = total_kl / total_kl_tokens if total_kl_tokens > 0 else 0.0
    return total_loss, mean_kl, n_uniform


@app.command()
def train(
    sft_checkpoint: str = typer.Option(..., "--sft-checkpoint", "-c",
        help="Path to SFT LoRA checkpoint (e.g. checkpoints/sft_qwen_1.5b/final)"),
    out: Path = typer.Option(Path("checkpoints/grpo_qwen_1.5b"), "--out", "-o"),
    train_questions: Path = typer.Option(
        Path("data/musique/questions/train_set.json"), "--train-questions"),
    corpus_path: str = typer.Option("data/musique/corpus", "--corpus"),
    steps: int = typer.Option(75, "--steps", help="Total gradient steps"),
    batch_size: int = typer.Option(4, "--batch-size",
        help="Distinct questions per gradient step. Higher = less variance, "
             "proportionally slower. Reduce --steps accordingly to keep total "
             "rollouts (and cost) roughly constant: steps * batch_size * group_size."),
    group_size: int = typer.Option(8, "--group-size",
        help="Rollouts per question. Higher = more reward variance = better signal"),
    lr: float = typer.Option(1e-6, "--lr"),
    max_episode_steps: int = typer.Option(10, "--max-episode-steps"),
    save_steps: int = typer.Option(15, "--save-steps"),
    temperature: float = typer.Option(1.0, "--temperature",
        help="Higher temperature encourages diverse rollouts"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit",
        help="Use 4-bit quantization (disable if PyTorch too old for bitsandbytes)"),
) -> None:
    """GRPO fine-tuning from the SFT checkpoint using the document exploration env."""
    console.print("[bold]GRPO Training — Document Exploration[/bold]\n")
    console.print(f"batch_size={batch_size}, group_size={group_size}, steps={steps}, lr={lr}, temp={temperature}")
    console.print(f"[dim]beta={KL_BETA} KL penalty vs frozen base, DR-GRPO normalization[/dim]\n")

    console.print("Loading model from SFT checkpoint...")
    model = _load_model(sft_checkpoint, trainable=True, load_in_4bit=load_in_4bit)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("Loading corpus...")
    corpus = Corpus(corpus_path=corpus_path)
    corpus.load()
    questions = json.loads(train_questions.read_text())
    console.print(f"Loaded {len(questions)} train questions\n")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    out.mkdir(parents=True, exist_ok=True)
    reward_window: list[float] = []
    total_uniform = 0

    for step in range(steps):
        q_indices = random.sample(range(len(questions)), min(batch_size, len(questions)))

        batch_step_data: list[list] = []
        batch_rewards: list[list[float]] = []

        for q_idx in q_indices:
            group_step_data: list[list] = []
            group_rewards: list[float] = []

            for _ in range(group_size):
                env = DocumentExplorationEnv(
                    corpus=corpus,
                    questions=questions,
                    max_steps=max_episode_steps,
                    use_docker=None,
                    corpus_path=corpus_path,
                )
                try:
                    sd, r = _collect_rollout(
                        model, tokenizer, env, questions, q_idx,
                        max_steps=max_episode_steps, temperature=temperature,
                    )
                    group_step_data.append(sd)
                    group_rewards.append(r)
                except Exception as e:
                    logger.warning(f"Rollout error: {e}")
                    group_step_data.append([])
                    group_rewards.append(0.0)
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            batch_step_data.append(group_step_data)
            batch_rewards.append(group_rewards)
            reward_window.extend(group_rewards)

        if len(reward_window) > 80:
            reward_window = reward_window[-80:]

        total_loss, mean_kl, n_uniform = _grpo_update(
            model, optimizer, batch_step_data, batch_rewards, step_num=step
        )
        total_uniform += n_uniform

        recent_avg = sum(reward_window) / len(reward_window)
        q_ids = [questions[i]["id"] for i in q_indices]
        logger.info(
            f"Step {step:03d} | {q_ids} | loss={total_loss:.4f} | kl={mean_kl:.4f} | "
            f"uniform={n_uniform}/{len(q_indices)} | recent_avg={recent_avg:.3f}"
        )

        if (step + 1) % save_steps == 0:
            ckpt = out / f"step_{step + 1}"
            model.save_pretrained(str(ckpt))
            logger.info(f"Checkpoint → {ckpt}")

    final = out / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    console.print(f"\n[bold green]Done. Saved to {final}[/bold green]")
    console.print(f"Total uniform-reward questions across all steps: {total_uniform}")


if __name__ == "__main__":
    app()
