"""QLoRA SFT on code-execution trajectories.

Requires the [training] extra:
  pip install -e ".[training]"

Run on a GPU box (single H100/A100/A10G):
  python scripts/train_sft.py train --data data/sft/qwen_traj_smoke.jsonl --epochs 1
  python scripts/train_sft.py train --data data/sft/qwen_traj_full.jsonl \
      --epochs 2 --out checkpoints/sft_qwen_1.5b

After training, sanity-check the checkpoint:
  python scripts/train_sft.py sanity-check --model checkpoints/sft_qwen_1.5b/final
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# r=4, not 16: with ~190 short-assistant-span examples the dataset carries far
# less information than even a rank-1 adapter can hold, so the binding risk is
# memorisation, not capacity. Lower rank is free regularisation here.
# alpha tracks r to hold the alpha/r update scale at 2 — leaving alpha at 32
# while dropping r would quadruple the effective step size on top of the LR.
LORA_CONFIG = {
    "r": 4,
    "lora_alpha": 8,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


def _expand_per_action(rows: list[dict]) -> list[dict]:
    """Turn each trajectory into one next-action example per assistant turn.

    Qwen3's non-thinking generation prefix is present only for the assistant turn
    being generated. A whole-conversation SFT row therefore gives intermediate
    actions a different prefix from inference. Splitting by action preserves the
    complete preceding history while making every target the final assistant turn.
    """
    examples: list[dict] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"row {row_index} has no conversational 'messages' list")
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt = messages[:message_index]
            if not prompt or prompt[-1].get("role") != "user":
                raise ValueError(
                    f"row {row_index}, message {message_index}: assistant target must "
                    "immediately follow a user observation"
                )
            examples.append({"prompt": prompt, "completion": [message]})
    if not examples:
        raise ValueError("dataset contains no assistant actions")
    return examples


def _tokenize_per_action(
    tokenizer, examples: list[dict], max_seq_len: int, split_name: str
) -> tuple[list[dict], dict]:
    """Create labels only for the next action and prove prefix equality.

    Building labels here avoids relying on trainer-version-specific chat-template
    preprocessing. The exact token prefix used for training must equal the prompt
    produced by Qwen3 at inference with thinking disabled.
    """
    tokenized: list[dict] = []
    lengths: list[int] = []
    supervised_tokens = 0

    def input_ids(rendered) -> list[int]:
        # transformers 4.x returns a list here; 5.x may return a
        # BatchEncoding when the template contains generation markers.
        return rendered["input_ids"] if hasattr(rendered, "keys") else rendered

    for row_index, example in enumerate(examples):
        prompt = example["prompt"]
        completion = example["completion"]
        prompt_ids = input_ids(tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
        full_ids = input_ids(tokenizer.apply_chat_template(
            prompt + completion,
            tokenize=True,
            enable_thinking=False,
        ))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{split_name} action {row_index}: training tokens do not start with "
                "the inference generation prefix"
            )
        if len(full_ids) > max_seq_len:
            raise ValueError(
                f"{split_name} action {row_index}: {len(full_ids)} tokens exceeds "
                f"--max-seq-len {max_seq_len}; refusing silent target truncation"
            )
        target_tokens = len(full_ids) - len(prompt_ids)
        if target_tokens <= 0:
            raise ValueError(f"{split_name} action {row_index}: empty supervised target")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        tokenized.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })
        lengths.append(len(full_ids))
        supervised_tokens += target_tokens

    return tokenized, {
        "actions": len(tokenized),
        "input_tokens": sum(lengths),
        "supervised_tokens": supervised_tokens,
        "max_tokens": max(lengths),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _load_training_deps():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
        return torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Run: pip install -e '.[training]'")
        sys.exit(1)


@app.command()
def train(
    data: Path = typer.Option(..., "--data", "-d", help="JSONL file from collect_sft_data.py"),
    val_data: Path = typer.Option(None, "--val-data",
        help="Held-out JSONL. Eval loss diverging from train loss means memorisation."),
    out: Path = typer.Option(Path("checkpoints/sft_qwen_1.5b"), "--out", "-o"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
    # LoRA needs a markedly higher LR than full fine-tuning; 2e-5 is a
    # full-fine-tune value and barely moves a rank-4 adapter.
    lr: float = typer.Option(2e-4, "--lr"),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
    max_seq_len: int = typer.Option(8192, "--max-seq-len"),
    base_model: str = typer.Option(BASE_MODEL, "--base-model"),
    base_revision: str = typer.Option(BASE_REVISION, "--base-revision"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    per_action: bool = typer.Option(
        True,
        "--per-action/--full-conversation",
        help="Supervise each next action with the exact inference prefix",
    ),
    max_steps: int = typer.Option(-1, "--max-steps"),
    save_steps: int = typer.Option(
        0, "--save-steps", help="Save every N optimizer steps; 0 saves per epoch"
    ),
    save_total_limit: int = typer.Option(10, "--save-total-limit"),
    seed: int = typer.Option(42, "--seed"),
    corpus_manifest: Path = typer.Option(
        None, "--corpus-manifest", help="Source-corpus manifest recorded with the run"
    ),
) -> None:
    torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer = _load_training_deps()

    # Load data
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    console.print(f"Loaded {len(rows)} conversations from {data}")

    val_rows = None
    if val_data:
        val_rows = [json.loads(line) for line in val_data.read_text().splitlines() if line.strip()]
        console.print(f"Loaded {len(val_rows)} held-out conversations from {val_data}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3's stock template has no {% generation %} block, so assistant_only_loss
    # would silently mask nothing and train on garbage. Raises if it can't verify.
    if "qwen3" in base_model.lower():
        from src.policies.qwen3_chat_template import patch_tokenizer_for_assistant_masking

        patch_tokenizer_for_assistant_masking(tokenizer)
        console.print("[green]Patched Qwen3 chat template for assistant-only loss[/green]")

    train_stats = {"conversations": len(rows)}
    val_stats = {"conversations": len(val_rows)} if val_rows is not None else None
    if per_action:
        action_rows = _expand_per_action(rows)
        tokenized_rows, action_stats = _tokenize_per_action(
            tokenizer, action_rows, max_seq_len, "train"
        )
        train_stats.update(action_stats)
        dataset = Dataset.from_list(tokenized_rows)
        eval_dataset = None
        if val_rows is not None:
            val_action_rows = _expand_per_action(val_rows)
            tokenized_val, val_action_stats = _tokenize_per_action(
                tokenizer, val_action_rows, max_seq_len, "validation"
            )
            val_stats.update(val_action_stats)
            eval_dataset = Dataset.from_list(tokenized_val)
        console.print(
            f"[green]Verified exact inference-prefix alignment for "
            f"{train_stats['actions']} training actions[/green]"
        )
    else:
        if "qwen3" in base_model.lower() and any(
            sum(m.get("role") == "assistant" for m in row.get("messages", [])) > 1
            for row in rows
        ):
            raise ValueError(
                "Qwen3 full-conversation training misaligns intermediate action prefixes. "
                "Use the default --per-action mode."
            )
        dataset = Dataset.from_list(rows)
        eval_dataset = Dataset.from_list(val_rows) if val_rows is not None else None

    # Model
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if not load_in_4bit else None,
    )

    lora_config = LoraConfig(**LORA_CONFIG)

    out_dir = str(out)
    final_dir = str(out / "final")

    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # Must be set explicitly: HF defaults eval batch to 8, which OOMs on a
        # 24GB card once sequences approach the 8k limit even though training at
        # batch 1 fits comfortably.
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        # Per-epoch, not save_steps: a short run can finish in fewer steps than
        # save_steps, in which case no intermediate checkpoint is ever written and
        # a crash loses everything. Saving per epoch also makes the epoch-1
        # checkpoint available if epoch-2 eval loss shows memorisation.
        save_strategy="steps" if save_steps > 0 else "epoch",
        save_steps=save_steps if save_steps > 0 else 500,
        save_total_limit=save_total_limit,
        packing=False,
        max_length=max_seq_len,
        report_to="none",
        gradient_checkpointing=True,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        # Without this, loss is computed over the entire sequence — including the
        # long tool-output text (search results, REPL prints) in user turns, which
        # the model never needs to generate. On our data only ~20% of tokens are
        # assistant tokens, so this is the difference between training on the
        # trajectory and training mostly on search output. Qwen3 needs the
        # chat-template patch applied above for this to mask anything at all.
        # Per-action rows are pre-tokenized above with explicit -100 prompt
        # labels. Conversational rows use the generation markers in the patched
        # template instead.
        assistant_only_loss=not per_action,
        completion_only_loss=False if per_action else None,
        max_steps=max_steps,
        seed=seed,
        data_seed=seed,
    )

    tokenizer.model_max_length = max_seq_len

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "qwen-sft-run-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "base_revision": base_revision,
        "training_format": "per-action-prefix-aligned" if per_action else "conversation",
        "data": {"path": str(data), "sha256": _sha256(data), **train_stats},
        "validation": (
            {"path": str(val_data), "sha256": _sha256(val_data), **val_stats}
            if val_data is not None and val_stats is not None else None
        ),
        "corpus_manifest": (
            {"path": str(corpus_manifest), "sha256": _sha256(corpus_manifest)}
            if corpus_manifest is not None else None
        ),
        "lora": LORA_CONFIG,
        "training_script_sha256": _sha256(Path(__file__)),
        "training": {
            "epochs": epochs, "max_steps": max_steps, "learning_rate": lr,
            "batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
            "max_sequence_length": max_seq_len, "load_in_4bit": load_in_4bit,
            "save_steps": save_steps, "save_total_limit": save_total_limit,
            "seed": seed,
        },
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--porcelain").splitlines(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    console.print(
        f"[bold]Training on {len(dataset)} action examples from {len(rows)} conversations, "
        f"{epochs} epoch(s)[/bold]"
    )
    trainer.train()
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"[green]Saved to {final_dir}[/green]")


@app.command()
def sanity_check(
    model: Path = typer.Option(..., "--model", "-m", help="Path to saved LoRA checkpoint"),
    question: str = typer.Option(
        "Where was the CEO of Apex Corp born?",
        "--question", "-q",
    ),
    max_new_tokens: int = typer.Option(512, "--max-tokens"),
) -> None:
    """Greedy-decode one question. Output should be Python code, not prose."""
    torch, _, LoraConfig, _, AutoModelForCausalLM, AutoTokenizer, _, _, _ = _load_training_deps()
    from peft import PeftModel

    base_model = BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base, str(model))
    peft_model.eval()

    _SYSTEM_PROMPT = (
        "You are an agent exploring a document corpus via Python code.\n\n"
        "Tools (already imported):\n"
        "  search(query, top_k=5)         → [{\"doc_id\", \"title\", \"chunk\", \"score\"}]\n"
        "  search(query, method=\"chunk\")  → chunk-level search for buried facts\n"
        "  read(doc_id)                   → full document text\n"
        "  extract(doc_id, regex)         → regex matches from a doc\n"
        "  search_within(doc_id, query)   → relevant windows inside a specific doc\n"
        "  verify(doc_id, claim)          → {\"found\", \"match_ratio\", \"excerpt\"}\n"
        "  list_docs()                    → [{\"doc_id\", \"title\", \"chars\"}]\n\n"
        "Each turn: write Python code OR a SUBMIT line. Never both. Never prose. Never markdown.\n"
        "Variables persist across turns. Use print() to see output.\n\n"
        "When you have the answer:\n"
        "SUBMIT: <your answer> CITATIONS: [\"doc_id_1\", \"doc_id_2\"]"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(peft_model.device)

    with torch.no_grad():
        out = peft_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"\n[bold]Model output:[/bold]\n{response}")

    if response.strip().upper().startswith("SUBMIT:") or any(
        kw in response for kw in ["search(", "read(", "extract(", "search_within("]
    ):
        console.print("\n[green]✓ Output looks like Python code or SUBMIT — format correct[/green]")
    else:
        console.print("\n[red]✗ Output looks like prose — check chat template or data[/red]")


if __name__ == "__main__":
    app()
