"""LoRA SFT on Qwen2.5-1.5B-Instruct using high-reward Claude trajectories.

Requires the [training] extra:
  pip install -e ".[training]"

Run on a GPU box (single H100/A100/A10G):
  python scripts/train_sft.py --data data/sft/qwen_traj_smoke.jsonl --epochs 1
  python scripts/train_sft.py --data data/sft/qwen_traj_full.jsonl --epochs 2 --out checkpoints/sft_qwen_1.5b

After training, sanity-check the checkpoint:
  python scripts/train_sft.py --sanity-check --model checkpoints/sft_qwen_1.5b/final
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


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
    out: Path = typer.Option(Path("checkpoints/sft_qwen_1.5b"), "--out", "-o"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
    lr: float = typer.Option(2e-5, "--lr"),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
    max_seq_len: int = typer.Option(8192, "--max-seq-len"),
    base_model: str = typer.Option(BASE_MODEL, "--base-model"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
) -> None:
    torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer = _load_training_deps()

    # Load data
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    console.print(f"Loaded {len(rows)} conversations from {data}")

    dataset = Dataset.from_list(rows)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        packing=False,
        report_to="none",
    )

    tokenizer.model_max_length = max_seq_len

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    console.print(f"[bold]Training on {len(dataset)} examples, {epochs} epoch(s)[/bold]")
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
        base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base, str(model))
    peft_model.eval()

    from src.env.verifiers_env import SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
