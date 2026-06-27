"""Custom GRPO training loop for document exploration.

Trains Qwen2.5-1.5B LoRA via GRPO using our DocumentExplorationEnv directly.
No verifiers dependency — rollouts go through the gym env, gradients via HF transformers.

Key design choices (informed by DAPO/DR-GRPO literature):
- group_size=8: more rollouts per question increases reward variance likelihood
- beta=0.0: no KL penalty — SFT reference is too weak to anchor to usefully
- Fixed token normalization: avoids length bias from per-episode normalization
- Skip groups where all rewards are identical (no learning signal)

Usage:
  python scripts/train_grpo_custom.py \
    --sft-checkpoint checkpoints/sft_qwen_1.5b/final \
    --out checkpoints/grpo_qwen_1.5b \
    --steps 300
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


def _collect_rollout(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    env: DocumentExplorationEnv,
    questions: list[dict],
    q_idx: int,
    max_steps: int = 10,
    temperature: float = 0.8,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
    """Run one episode. Returns (step_data, reward).

    step_data: list of (context_ids, action_ids) tensors (CPU) per assistant turn.
    Higher temperature (0.8) encourages more diverse rollouts so reward varies within a group.
    """
    q = questions[q_idx]
    env.reset(question_idx=q_idx)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {q['question']}"},
    ]

    step_data: list[tuple[torch.Tensor, torch.Tensor]] = []
    total_reward = 0.0

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            ctx_ids = enc.input_ids

            gen_ids = model.generate(
                ctx_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

            action_ids = gen_ids[0][ctx_ids.shape[1]:].cpu()
            step_data.append((ctx_ids[0].cpu(), action_ids))

            action = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": action})

            obs, step_reward, done, _ = env.step(action)
            total_reward += step_reward  # accumulates: hit bonuses + final SUBMIT reward
            if done:
                break
            messages.append({"role": "user", "content": obs})

    model.train()
    return step_data, total_reward


def _step_log_prob(
    model: PeftModel, ctx_ids: torch.Tensor, action_ids: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Log prob sum for one (context, action) pair.

    Deletes the full logits tensor immediately after slicing — Qwen2.5 vocab is
    152k so keeping it in the graph wastes ~1GB per forward pass.
    """
    if action_ids.numel() == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True), 0

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
    return token_lp.sum(), act_len


def _grpo_update(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    group_step_data: list[list[tuple[torch.Tensor, torch.Tensor]]],
    group_rewards: list[float],
) -> float:
    """GRPO gradient accumulation — backward after each (rollout, step) to avoid OOM.

    With 7B + 152k vocab, holding all 8×10 computation graphs simultaneously
    needs ~80GB. Backward per step frees each graph immediately; peak memory
    drops to 1 forward pass. Gradients accumulate in param.grad across calls.
    """
    rewards = torch.tensor(group_rewards, dtype=torch.float32)
    advantages = ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()

    n_valid = sum(1 for sd in group_step_data if sd)
    if n_valid == 0:
        return 0.0

    optimizer.zero_grad()
    total_loss = 0.0

    for step_data, adv in zip(group_step_data, advantages):
        if not step_data:
            continue
        for ctx_ids, action_ids in step_data:
            lp, n_tok = _step_log_prob(model, ctx_ids, action_ids)
            if n_tok == 0:
                continue
            piece = -adv * lp / NORM_TOKENS / n_valid
            piece.backward()
            total_loss += piece.item()
            del lp, piece
            torch.cuda.empty_cache()

    return total_loss


@app.command()
def train(
    sft_checkpoint: str = typer.Option(..., "--sft-checkpoint", "-c",
        help="Path to SFT LoRA checkpoint (e.g. checkpoints/sft_qwen_1.5b/final)"),
    out: Path = typer.Option(Path("checkpoints/grpo_qwen_1.5b"), "--out", "-o"),
    train_questions: Path = typer.Option(
        Path("data/musique/questions/train_set.json"), "--train-questions"),
    corpus_path: str = typer.Option("data/musique/corpus", "--corpus"),
    steps: int = typer.Option(300, "--steps", help="Total gradient steps"),
    group_size: int = typer.Option(8, "--group-size",
        help="Rollouts per question. Higher = more reward variance = better signal"),
    lr: float = typer.Option(1e-6, "--lr"),
    max_episode_steps: int = typer.Option(10, "--max-episode-steps"),
    save_steps: int = typer.Option(50, "--save-steps"),
    temperature: float = typer.Option(1.0, "--temperature",
        help="Higher temperature encourages diverse rollouts"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit",
        help="Use 4-bit quantization (disable if PyTorch too old for bitsandbytes)"),
) -> None:
    """GRPO fine-tuning from the SFT checkpoint using the document exploration env."""
    console.print("[bold]GRPO Training — Document Exploration[/bold]\n")
    console.print(f"group_size={group_size}, steps={steps}, lr={lr}, temp={temperature}")
    console.print("[dim]beta=0 (no KL), DR-GRPO normalization, skip uniform-reward groups[/dim]\n")

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
    skipped = 0

    for step in range(steps):
        q_idx = random.randint(0, len(questions) - 1)
        q_id = questions[q_idx]["id"]

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

        reward_window.extend(group_rewards)
        if len(reward_window) > 80:
            reward_window = reward_window[-80:]

        # Skip if all rewards identical — group normalization gives zero advantage
        if len({round(r, 4) for r in group_rewards}) == 1:
            skipped += 1
            logger.info(
                f"Step {step:03d} | {q_id} | uniform reward={group_rewards[0]:.3f} "
                f"— skip ({skipped} total skips)"
            )
            continue

        total_loss = _grpo_update(model, optimizer, group_step_data, group_rewards)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        recent_avg = sum(reward_window) / len(reward_window)
        logger.info(
            f"Step {step:03d} | {q_id} | loss={total_loss:.4f} | "
            f"rewards={[f'{r:.3f}' for r in group_rewards]} | "
            f"recent_avg={recent_avg:.3f}"
        )

        if (step + 1) % save_steps == 0:
            ckpt = out / f"step_{step + 1}"
            model.save_pretrained(str(ckpt))
            logger.info(f"Checkpoint → {ckpt}")

    final = out / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    console.print(f"\n[bold green]Done. Saved to {final}[/bold green]")
    console.print(f"Total skipped steps (uniform reward): {skipped}/{steps}")


if __name__ == "__main__":
    app()
