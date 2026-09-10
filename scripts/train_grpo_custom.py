"""Custom GRPO training loop for document exploration.

Trains Qwen2.5-7B LoRA via GRPO using our DocumentExplorationEnv directly.
No verifiers dependency — rollouts go through the gym env, gradients via HF transformers.

Key design choices (informed by DAPO/DR-GRPO/Search-R1 literature):
- batch_size=4 distinct questions per gradient step, group_size=8 rollouts each:
  literature batches 64-512 questions per step to avoid "ping-pong" gradient
  variance from single-question updates; 4 is a cost-feasible middle ground.
  Reduce --steps when raising batch_size to keep total rollouts (cost) similar.
- beta=0.001 KL penalty with a selectable pre-SFT base or frozen-SFT reference;
  the historical default remains the base reference for controlled comparisons.
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

from contextlib import contextmanager
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import typer

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.policies.qwen_common import SYSTEM_PROMPT

if TYPE_CHECKING:
    from peft import PeftModel

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# PEFT adapter names: the trainable policy, and a frozen SFT copy used as the
# KL reference when --kl-ref=sft.
POLICY_ADAPTER = "default"
REF_ADAPTER = "kl_ref"

# Fixed normalization constant — avoids length bias (DR-GRPO style)
NORM_TOKENS = 256
# Shared rollout/scoring context budget. Actual peak GPU memory needs measurement.
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


# The system prompt carries every tool signature (search/read/extract/
# search_within/verify/list_docs). Measured once at startup so truncation can
# preserve it. Set by _set_prompt_head_len().
_PROMPT_HEAD_TOKENS = 0
_trunc_seen = 0
_trunc_hit = 0


def _set_prompt_head_len(tokenizer: AutoTokenizer) -> None:
    """Measure the system-prompt prefix so _prepare_ctx can keep it."""
    global _PROMPT_HEAD_TOKENS
    _PROMPT_HEAD_TOKENS = len(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}],
            tokenize=True, add_generation_prompt=False,
        )
    )
    logger.info(f"Reserving {_PROMPT_HEAD_TOKENS} head tokens (system prompt) from truncation")


def _prepare_ctx(ctx_ids: torch.Tensor, head_tokens: int | None = None) -> torch.Tensor:
    """Bound context length while preserving the system prompt.

    Naive left-truncation (ctx_ids[-MAX_CTX_TOKENS:]) drops the HEAD, which is
    exactly where the tool documentation lives. A read() of a 5k-char document is
    ~1.2k tokens and MAX_OUTPUT_CHARS=8000 permits ~2k, so a multi-turn episode
    exceeds MAX_CTX_TOKENS within a turn or two — after which the model would be
    trained to emit tool calls whose definitions are no longer in its context.

    Drop from the MIDDLE instead: keep the system-prompt head plus the most recent
    tokens. Called BEFORE generation; scoring uses the exact stored result.
    Rollouts pass the complete system + question prefix as head_tokens.
    """
    global _trunc_seen, _trunc_hit
    _trunc_seen += 1
    if ctx_ids.shape[0] <= MAX_CTX_TOKENS:
        return ctx_ids
    _trunc_hit += 1
    head = _PROMPT_HEAD_TOKENS if head_tokens is None else head_tokens
    if not 0 <= head < MAX_CTX_TOKENS:
        raise ValueError("System prompt and question must leave room within MAX_CTX_TOKENS")
    if head <= 0:
        return ctx_ids[-MAX_CTX_TOKENS:]
    return torch.cat([ctx_ids[:head], ctx_ids[-(MAX_CTX_TOKENS - head):]])


def _scoring_context(ctx_ids: torch.Tensor) -> torch.Tensor:
    """Reject mismatched legacy rollouts instead of silently changing context."""
    if ctx_ids.ndim != 1 or not 0 < ctx_ids.numel() <= MAX_CTX_TOKENS:
        raise ValueError("Score the bounded context stored at generation time")
    return ctx_ids


def _action_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    return F.log_softmax(logits.float() / temperature, dim=-1)


def _truncation_rate() -> float:
    return (_trunc_hit / _trunc_seen) if _trunc_seen else 0.0


@torch.no_grad()
def _compute_token_log_probs(
    model: Any, ctx_ids: torch.Tensor, action_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Compute exact per-token log-probs for action tokens given context.

    Matches the rollout's temperature-scaled, unfiltered sampling distribution.
    """
    if action_ids.numel() == 0:
        return torch.empty(0)

    device = next(model.parameters()).device
    ctx = _scoring_context(ctx_ids).to(device)
    act = action_ids.to(device)
    ctx_len = ctx.shape[0]
    act_len = act.shape[0]

    full_ids = torch.cat([ctx, act]).unsqueeze(0)
    logits = model(full_ids).logits[0]
    act_logits = logits[ctx_len - 1 : ctx_len - 1 + act_len]
    act_log_probs = _action_log_probs(act_logits, temperature)
    token_lp = act_log_probs.gather(1, act.unsqueeze(1)).squeeze(1)
    return token_lp.detach().cpu()


def _collect_rollout(
    model: Any,
    tokenizer: AutoTokenizer,
    env: DocumentExplorationEnv,
    questions: list[dict],
    q_idx: int,
    max_steps: int = 10,
    temperature: float = 1.0,
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
    head_tokens = len(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    ))
    if head_tokens >= MAX_CTX_TOKENS:
        raise ValueError("System prompt and question exceed the rollout context budget")
    # A fresh config prevents inherited top-k/top-p/repetition processors from
    # changing the distribution relative to the log-probability calculation.
    generation_config = GenerationConfig(
        max_new_tokens=NORM_TOKENS, do_sample=True, temperature=temperature,
        top_k=0, top_p=1.0, repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    step_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_reward = 0.0

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            ctx_ids = _prepare_ctx(enc.input_ids[0], head_tokens=head_tokens)
            ctx_ids = ctx_ids.unsqueeze(0).to(next(model.parameters()).device)

            gen_out = model.generate(
                ctx_ids,
                attention_mask=torch.ones_like(ctx_ids),
                generation_config=generation_config,
            )

            action_ids = gen_out.sequences[0][ctx_ids.shape[1]:].cpu()
            if action_ids.numel() > 0:
                old_token_lp = _compute_token_log_probs(
                    model, ctx_ids[0], action_ids, temperature=temperature,
                )
            else:
                old_token_lp = torch.empty(0)
            step_data.append((ctx_ids[0].cpu(), action_ids, old_token_lp))

            action = tokenizer.decode(action_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": action})

            obs, step_reward, done, _ = env.step(action)
            total_reward += step_reward
            if done:
                break
            messages.append({"role": "user", "content": obs})

    model.train()
    return step_data, total_reward


# KL penalty holding the policy near a reference. WHICH reference matters:
#
#   "base" — adapter disabled, i.e. the PRE-SFT base model. Training starts from
#            the SFT adapter, so KL(policy||base) is already nonzero at step 0 and
#            its gradient points AWAY from the SFT solution. It does not regularise
#            drift from your initialisation; it partially undoes it. Format
#            competence came from SFT (RESULTS.md 4.3: SFT "immediately writes
#            search()", base emitted prose and SyntaxErrors), so pulling toward
#            base pulls toward malformed output, not away from it.
#
#   "sft"  — a frozen copy of the SFT adapter. Standard RLVR practice: anchor to
#            the checkpoint RL started from.
#
# The effect of beta=0.001 has not been measured for this trainer. A larger value
# would give the chosen reference more influence: the Valyu router paper
# (arXiv:2608.00030), for example, reports beta=0.15 to discourage drift from SFT
# behaviour. That motivates an explicit guard against accidentally applying a
# strong penalty toward the pre-SFT base; it does not establish a universal safe
# threshold for this task.
KL_BETA = 0.001

# Conservative operator guardrail, pending a measured beta/reference ablation.
KL_BETA_BASE_MAX = 0.01


@contextmanager
def _reference_adapter(model: Any, kl_ref: str):
    """Activate the KL reference policy for the duration of the block.

    "base" disables the adapter (pre-SFT weights). "sft" activates a frozen copy
    of the SFT adapter loaded under the name REF_ADAPTER.
    """
    if kl_ref == "base":
        with model.disable_adapter():
            yield
        return

    active = getattr(model, "active_adapters", None) or getattr(model, "active_adapter", None)
    previous = list(active) if isinstance(active, (list, tuple)) else [active or POLICY_ADAPTER]
    model.set_adapter(REF_ADAPTER)
    try:
        yield
    finally:
        model.set_adapter(previous[0] if len(previous) == 1 else previous)


def _step_log_prob_and_kl(
    model: Any, ctx_ids: torch.Tensor, action_ids: torch.Tensor, temperature: float = 1.0,
    kl_ref: str = "base",
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
    ctx = _scoring_context(ctx_ids).to(device)
    act = action_ids.to(device)

    ctx_len = ctx.shape[0]
    act_len = act.shape[0]

    full_ids = torch.cat([ctx, act]).unsqueeze(0)
    logits = model(full_ids).logits[0]
    act_logits = logits[ctx_len - 1 : ctx_len - 1 + act_len].clone()
    del logits
    torch.cuda.empty_cache()

    act_log_probs = _action_log_probs(act_logits, temperature)
    token_lp = act_log_probs.gather(1, act.unsqueeze(1)).squeeze(1)

    with torch.no_grad(), _reference_adapter(model, kl_ref):
        ref_logits = model(full_ids).logits[0]
        ref_act_logits = ref_logits[ctx_len - 1 : ctx_len - 1 + act_len]
        ref_log_probs = _action_log_probs(ref_act_logits, temperature)
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
    temperature: float = 1.0,
    kl_ref: str = "base",
) -> tuple[float, float, int, list[dict[str, float]]]:
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
    for group_data, group_rewards in zip(batch_step_data, batch_rewards):
        if len(group_data) != len(group_rewards) or any(not sd for sd in group_data):
            raise ValueError("Incomplete rollout group: do not treat execution failures as rewards")
        if len(group_rewards) < 2 or not all(torch.isfinite(torch.tensor(r)) for r in group_rewards):
            raise ValueError("Each rollout group needs at least two finite rewards")
        if len({round(r, 4) for r in group_rewards}) == 1:
            n_uniform += 1
        rewards = torch.tensor(group_rewards, dtype=torch.float32)
        batch_advantages.append(
            ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()
        )

    total_loss = 0.0
    total_kl = 0.0
    total_kl_tokens = 0

    epoch_stats: list[dict[str, float]] = []

    for _epoch in range(PPO_EPOCHS):
        optimizer.zero_grad()
        total_loss = 0.0
        total_kl = 0.0
        total_kl_tokens = 0
        n_clipped = 0
        n_ratio_tokens = 0

        for group_step_data, advantages in zip(batch_step_data, batch_advantages):
            n_valid = sum(1 for sd in group_step_data if sd)
            if n_valid == 0:
                continue

            for step_data, adv in zip(group_step_data, advantages):
                if not step_data:
                    continue
                for ctx_ids, action_ids, old_token_lp in step_data:
                    token_lp, kl_per_token, n_tok = _step_log_prob_and_kl(
                        model, ctx_ids, action_ids, temperature=temperature, kl_ref=kl_ref,
                    )
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
                    with torch.no_grad():
                        n_clipped += int((
                            (ratio < 1.0 - PPO_CLIP_EPS) | (ratio > 1.0 + PPO_CLIP_EPS)
                        ).sum().item())
                        n_ratio_tokens += ratio.numel()
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

        # clip_grad_norm_ RETURNS the norm before clipping. Per-token summation
        # multiplied the raw loss by ~n_tokens. AdamW is invariant to a uniform
        # loss rescale, but this clip is not: if the norm sits above max_norm every
        # step, updates are clip-dominated and Adam's second-moment estimate is
        # flattened. This number is how you detect that.
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        ))
        epoch_stats.append({
            "kl": total_kl / total_kl_tokens if total_kl_tokens else 0.0,
            "clip_frac": n_clipped / n_ratio_tokens if n_ratio_tokens else 0.0,
            "grad_norm": gnorm,
        })
        optimizer.step()

    mean_kl = total_kl / total_kl_tokens if total_kl_tokens > 0 else 0.0
    return total_loss, mean_kl, n_uniform, epoch_stats


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
    kl_ref: str = typer.Option("base", "--kl-ref",
        help="KL reference policy: 'base' (pre-SFT weights) or 'sft' (frozen SFT copy). "
             "'sft' is standard RLVR practice; 'base' is the historical default here."),
    temperature: float = typer.Option(1.0, "--temperature",
        help="Higher temperature encourages diverse rollouts"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit",
        help="Use 4-bit quantization (disable if PyTorch too old for bitsandbytes)"),
) -> None:
    """GRPO fine-tuning from the SFT checkpoint using the document exploration env."""
    if group_size < 2 or batch_size < 1 or steps < 1 or save_steps < 1:
        raise typer.BadParameter("group-size must be >= 2; batch-size, steps, save-steps >= 1")
    if not 0 < temperature < float("inf"):
        raise typer.BadParameter("temperature must be finite and positive")
    if kl_ref not in ("base", "sft"):
        raise typer.BadParameter("--kl-ref must be 'base' or 'sft'")
    if kl_ref == "base" and KL_BETA > KL_BETA_BASE_MAX:
        raise typer.BadParameter(
            f"KL_BETA={KL_BETA} with --kl-ref=base exceeds the configured guardrail. "
            "A strong base-reference penalty can oppose the SFT warm-start. "
            f"Use --kl-ref=sft, or set KL_BETA <= {KL_BETA_BASE_MAX}."
        )
    console.print("[bold]GRPO Training — Document Exploration[/bold]\n")
    console.print(f"batch_size={batch_size}, group_size={group_size}, steps={steps}, lr={lr}, temp={temperature}")
    reference_label = "frozen SFT adapter" if kl_ref == "sft" else "pre-SFT base weights"
    console.print(
        f"[dim]beta={KL_BETA} KL penalty vs {reference_label}, "
        "DR-GRPO normalization[/dim]\n"
    )

    console.print("Loading model from SFT checkpoint...")
    model = _load_model(sft_checkpoint, trainable=True, load_in_4bit=load_in_4bit)
    if kl_ref == "sft":
        # Frozen second copy of the SFT adapter. An r=16 adapter is tens of MB
        # against a ~5GB 4-bit base, so this is far cheaper than a second model.
        model.load_adapter(sft_checkpoint, adapter_name=REF_ADAPTER, is_trainable=False)
        model.set_adapter(POLICY_ADAPTER)
        logger.info(f"KL reference: frozen SFT adapter '{REF_ADAPTER}'")
    else:
        logger.info("KL reference: pre-SFT base weights (adapter disabled)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Must run before any rollout: _prepare_ctx needs this to know how much of
    # the head (the tool documentation) to protect from truncation.
    _set_prompt_head_len(tokenizer)
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
                    # Infrastructure failures are not task failures. Abort before
                    # corrupting this group's advantage baseline with fake zeros.
                    logger.error(f"Rollout error: {e}")
                    raise
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

        total_loss, mean_kl, n_uniform, epoch_stats = _grpo_update(
            model, optimizer, batch_step_data, batch_rewards, step_num=step,
            temperature=temperature,
            kl_ref=kl_ref,
        )
        total_uniform += n_uniform

        recent_avg = sum(reward_window) / len(reward_window)
        q_ids = [questions[i]["id"] for i in q_indices]
        logger.info(
            f"Step {step:03d} | {q_ids} | loss={total_loss:.4f} | kl={mean_kl:.4f} | "
            f"uniform={n_uniform}/{len(q_indices)} | recent_avg={recent_avg:.3f}"
        )
        # The four numbers that explain a flat curve:
        #   uniform : groups with zero reward variance give no advantage gradient,
        #             no matter how long training runs
        #   clip    : per epoch; high in epochs 2-3 means rollout reuse buys nothing
        #   kl      : per epoch; flat KL means the policy is not moving at all
        #   |g|     : pre-clip grad norm; pinned above 1.0 means clip-dominated
        #   trunc   : share of contexts exceeding MAX_CTX_TOKENS
        if epoch_stats:
            logger.info(
                "          "
                + " ".join(
                    f"e{i}[kl={s['kl']:.4f} clip={s['clip_frac']:.2f} |g|={s['grad_norm']:.2f}]"
                    for i, s in enumerate(epoch_stats)
                )
                + f" | trunc={_truncation_rate():.2f}"
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
