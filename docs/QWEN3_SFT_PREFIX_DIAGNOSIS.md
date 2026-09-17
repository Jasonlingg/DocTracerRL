# Qwen3 SFT action-prefix mismatch

Audited September 17, 2026, after archiving the two QASPER SFT checkpoints in
`jasonlingg/rlm-explorer-qwen3-8b-qasper-sft` (private), revision
`7cbe6289760d7718886fe57e3a6f4c45905a5704`.

## Finding

The tokenizer saved with the actual training checkpoint formats intermediate
assistant actions differently from the final assistant action. Every retained
training conversation ends in `SUBMIT:`, while all preceding assistant actions
are Python. When rendered as one full conversation:

```text
<|im_start|>assistant
print(search('nodes'))<|im_end|>
...
<|im_start|>assistant
<think>

</think>

SUBMIT: 645 CITATIONS: ["d1"]<|im_end|>
```

`BaseQwenPolicy.act()` uses `add_generation_prompt=True, enable_thinking=False`.
That appends the empty `<think>...</think>` prefix on **every new action**,
including the first action before any evidence has been retrieved.

The full-conversation training representation therefore associates this prefix
exclusively with submission. At inference, even a code-writing decision is
presented with that prefix. This is a verified input-format mismatch and a
strong candidate explanation for premature submission. It is not yet a causal
demonstration that this is the only cause of the behavioral failure.

## CPU audit of the actual data and saved tokenizer

| Check | Training | Validation |
|---|---:|---:|
| Conversations | 189 | 47 |
| Code actions | 1,315 | 293 |
| Code actions preceded by the empty thinking block | 0 | 0 |
| Submission actions | 189 | 47 |
| Submission actions preceded by the empty thinking block | 189 | 47 |
| Code targets whose complete prefix differs from inference | 1,315 | 293 |
| Submission targets whose complete prefix differs from inference | 0 | 0 |
| Examples over the 8,192-token limit | 0 | 0 |
| Longest sequence | 6,882 | 7,011 |
| Rendered tokens | 684,271 | 167,292 |
| Assistant-mask tokens | 124,899 | 35,803 |

All system prompts match the current inference system prompt. Training token
count 684,271 also matches the first epoch's recorded trainer token count.
This audit did not instantiate TRL's collator or execute a model forward pass.

The validation set has the same format confound. Improving validation loss
therefore does not disprove this failure mechanism. It measures token prediction
under the training representation, not action selection under inference inputs.

The existing template tests checked that adding generation markers preserved
the stock full-conversation rendering and masked user text. They did **not**
compare the prefix of each supervised action with the prompt used when asking
the model to generate that action.

Audit artifact (ignored by Git):
`out/diagnostics/qwen3-training-prefix-audit.json`.

Identifiers:

- Saved training-template SHA-256:
  `58f5a8a0d1da685b8c61c1ef08931949568192ae38e873132b4df91e8d4255ec`
- Training JSONL SHA-256:
  `84a9f6403f288efa71775c10aac1945d89c5e489c7bf8cf788b698f42294d55f`
- Validation JSONL SHA-256:
  `4171ce7bc0cbc76c640a982d5802b29592553c91d08ee1c9351182a7f8aabb7c`

Minimal reproduction using the downloaded checkpoint tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/private/tmp/rlm-hf-export/epoch-1", local_files_only=True
)
messages = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Question: how many nodes?"},
    {"role": "assistant", "content": "print(search('nodes'))"},
    {"role": "user", "content": "Search found document d1."},
    {"role": "assistant", "content": 'SUBMIT: 645 CITATIONS: ["d1"]'},
]
print(tokenizer.apply_chat_template(messages, tokenize=False))
print(tokenizer.apply_chat_template(
    messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
))
```

## Next intervention and decision rule

Do not interpret an epoch-1 recovery alone as proof that two epochs caused
overfitting. It could instead reflect weaker learning of the same format
confound. The earlier proposed epoch-1 smoke remains useful for checkpoint
selection, but it cannot determine the root cause on its own.

The focused repair to evaluate is to supervise each next action against the
exact history and generation prefix used at inference. One implementation is
one prompt/completion example per assistant turn, retaining the preceding
multi-turn history and computing loss only on the target action. Preserve
the existing train/validation split; do not randomly split individual turns.
Pin non-thinking behavior, verify token-prefix equality, and check the actual
post-collation labels for both code and submission actions. TRL documents this
dataset format and completion-only loss in its
[SFT guide](https://huggingface.co/docs/trl/v1.3.0/en/sft_trainer#train-on-completion-only).

**Hypothesis:** aligning action prefixes removes the accidental association
between the inference prefix and terminal submission.

**Expected signal:** a small pilot with a fresh adapter on the same base model
recovers executable evidence-gathering actions before submission. Loss alone is
not a pass criterion. Converting trajectories into turn examples changes the
number of rows, so record optimizer steps and supervised tokens rather than
treating the old and new epoch counts as equivalent budgets.

**Decision rule:** first require exact training/inference prefix alignment and
correct target masks for all prepared examples. Then run a bounded training
pilot with early behavioral checks on a fixed development set. Use at least
4/5 episodes executing a research tool before submission as a plumbing gate,
and inspect whether those actions actually retrieve useful evidence. If that
fails, audit rendered inputs, raw generations, and action parsing before any
longer training run. Passing tool use does not establish answer quality; a
subsequent paired evaluation must establish supported-answer and abstention
performance.

The already-inspected five test questions are diagnostic examples now, not
fresh held-out confirmation. Keep final evaluation data separate from the
development loop and disclose previous test inspection.

## Implemented repair

`scripts/train_sft.py` now defaults to per-action supervision. It expands each
trajectory into one example for every assistant action, retains the complete
history before that action, and constructs labels only for the target action.
Before loading the model, it verifies at token level that every training sequence
starts with the exact non-thinking generation prefix used at inference. It refuses
over-length examples instead of silently truncating a target. Qwen3 multi-turn
training through the old `--full-conversation` representation is blocked.

The first corrected run intentionally keeps Qwen3-8B, rank-4 QLoRA, the existing
train/validation trajectory split, learning rate, batch size, and seed fixed. It
saves frequent adapters because converting 189 trajectories to 1,504 next-action
examples changes the optimizer-step count even though the target action text is
the same.

## Corrected run configuration

Launched September 17, 2026 on the existing RunPod Secure L4 pod. The run uses:

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`;
- the unchanged 189/47 trajectory split and unchanged rank-4 QLoRA target modules;
- 1,504 training actions and 340 validation actions;
- 124,899 training target tokens and 35,803 validation target tokens, exactly
  matching the assistant-token totals in the failed run;
- two epochs, learning rate `2e-4`, seed 42, batch size 1, and gradient
  accumulation 32;
- about 94 optimizer updates, close to the failed run's 96, with checkpoints
  every eight updates.

The larger gradient-accumulation value compensates for changing the row unit from
one trajectory to one action. It keeps the update count and learning-rate schedule
close to the control while the repeated history raises total processed input from
684,271 to 3,404,896 tokens per training epoch. A one-update GPU smoke completed a
real forward pass, backward pass, optimizer update, and adapter save before the
full run was launched.
