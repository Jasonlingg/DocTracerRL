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
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
import typer
from loguru import logger
from peft import PeftModel, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.policies.qwen_common import SYSTEM_PROMPT

console = Console()
app = typer.Typer()

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


def _load_model(checkpoint: str, trainable: bool, load_in_4bit: bool = True) -> PeftModel:
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


def _old_log_prob_from_scores(scores: tuple[torch.Tensor, ...], action_ids: torch.Tensor) -> float:
    """Sum log-prob of the sampled tokens from generate()'s returned scores.

    At temperature=1.0 (this script's default), these scores are mathematically
    identical to raw model logits — safe to use directly as the "old policy"
    log-prob for the PPO ratio below, with zero extra forward passes versus
    recomputing from scratch. If temperature is ever changed from 1.0, this
    assumption breaks and old-log-prob would need to come from a fresh forward
    pass instead.
    """
    n = min(len(scores), action_ids.shape[0])
    total = 0.0
    for i in range(n):
        log_probs = F.log_softmax(scores[i][0], dim=-1)
        total += log_probs[action_ids[i]].item()
    return total


def _collect_rollout(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    env: DocumentExplorationEnv,
    questions: list[dict],
    q_idx: int,
    max_steps: int = 10,
    temperature: float = 0.8,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, float]], float]:
    """Run one episode. Returns (step_data, reward).

    step_data: list of (context_ids, action_ids, old_log_prob) per assistant turn,
    where old_log_prob is captured at generation time for the PPO-clip ratio used
    when a batch of rollouts is reused across multiple gradient updates.
    Higher temperature (0.8) encourages more diverse rollouts so reward varies within a group.
    """
    q = questions[q_idx]
    env.reset(question_idx=q_idx)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {q['question']}"},
    ]

    step_data: list[tuple[torch.Tensor, torch.Tensor, float]] = []
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
                output_scores=True,
                return_dict_in_generate=True,
            )

            action_ids = gen_out.sequences[0][ctx_ids.shape[1]:].cpu()
            old_lp = _old_log_prob_from_scores(gen_out.scores, action_ids)
            step_data.append((ctx_ids[0].cpu(), action_ids, old_lp))

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
    model: PeftModel, ctx_ids: torch.Tensor, action_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Log prob sum and KL-vs-reference sum for one (context, action) pair.

    Reference log-probs come from the same model with the LoRA adapter disabled
    (frozen base weights) — avoids loading a second full model copy. Uses the k3
    KL estimator (always >= 0, low variance) from the GRPO/DeepSeekMath formulation.

    Deletes the full logits tensor immediately after slicing — Qwen2.5 vocab is
    152k so keeping it in the graph wastes ~1GB per forward pass.
    """
    if action_ids.numel() == 0:
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, torch.tensor(0.0, device=device), 0

    device = next(model.parameters()).device
    ctx = ctx_ids.to(device)
    act = action_ids.to(device)

    # Truncate context from the left to bound backward memory
    if ctx.shape[0] > MAX_CTX_TOKENS:
        ctx = ctx[-MAX_CTX_TOKENS:]

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

    return token_lp.sum(), kl_per_token.sum(), act_len


# PPO-clip range and epoch count — reusing each batch of (expensive-to-generate)
# rollouts for multiple gradient updates instead of one. Generation (multi-turn
# LLM calls + REPL execution) is the real bottleneck here, not backward passes,
# so this multiplies gradient signal per rollout at near-zero extra cost. After
# epoch 1 the policy has moved, so later epochs use an importance-sampling ratio
# against the generation-time ("old") policy, clipped to keep updates conservative
# — standard PPO/DAPO mechanics, just applied at the per-turn (not per-token) level
# to match this codebase's existing per-turn-summed log-probs.
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 3


def _grpo_update(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    batch_step_data: list[list[list[tuple[torch.Tensor, torch.Tensor, float]]]],
    batch_rewards: list[list[float]],
) -> tuple[float, float, int]:
    """GRPO gradient accumulation across a BATCH of questions, PPO_EPOCHS times.

    batch_step_data[i] / batch_rewards[i] are the group_size rollouts for the i-th
    question in the batch. Advantages are normalized WITHIN each question's own
    group (GRPO requirement) — but all questions' gradients accumulate into a
    single optimizer.step() per epoch, so one update reflects signal from multiple
    distinct questions instead of just one. This is what fixes the "ping-pong"
    variance of single-question-per-step training documented in the literature
    (Search-R1 etc. batch 64-512 questions per step; we use a small batch for cost
    reasons).

    With 7B + 152k vocab, holding all computation graphs simultaneously is
    infeasible. Backward per (question, rollout, step) frees each graph
    immediately; peak memory stays at ~1 forward pass regardless of batch_size.

    A question whose group has uniform reward contributes zero advantage-driven
    gradient automatically (numerator is 0) — no need to skip it outright, and
    skipping would also skip its KL term, which should still regularize the policy.

    Returns (last_epoch_loss, last_epoch_mean_kl, n_uniform_questions) for logging.
    """
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
                for ctx_ids, action_ids, old_lp in step_data:
                    lp, kl, n_tok = _step_log_prob_and_kl(model, ctx_ids, action_ids)
                    if n_tok == 0:
                        continue

                    ratio = torch.exp(lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    piece = (policy_loss + KL_BETA * kl) / NORM_TOKENS / n_valid / batch_n
                    piece.backward()
                    total_loss += piece.item()
                    total_kl += kl.item()
                    total_kl_tokens += n_tok
                    del lp, kl, piece, ratio, surr1, surr2, policy_loss
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

        total_loss, mean_kl, n_uniform = _grpo_update(model, optimizer, batch_step_data, batch_rewards)
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
