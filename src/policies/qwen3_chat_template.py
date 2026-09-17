"""Patch Qwen3's chat template so assistant-only loss masking actually works.

Qwen3's stock chat template contains no `{% generation %}` block. Without it,
`apply_chat_template(..., return_assistant_tokens_mask=True)` returns a mask of
all zeros and only emits a warning — so `assistant_only_loss=True` silently
trains on nothing (or on the whole sequence, including the long tool-output text
in user turns). The training run completes "successfully" and produces a garbage
checkpoint.

This wraps the assistant content and its <|im_end|> in generation markers,
leaving the rendered text byte-identical so training format still matches what
`BaseQwenPolicy.act()` produces at inference. The patch verifies both properties
and raises if either fails, so a future upstream template change breaks loudly
instead of silently.
"""

from __future__ import annotations

_THINK_BRANCH = (
    """{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n')"""
    """ + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"""
)
_THINK_BRANCH_PATCHED = (
    """{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n')"""
    """ + '\\n</think>\\n\\n' }}{% generation %}{{- content.lstrip('\\n') }}{% endgeneration %}"""
)

_PLAIN_BRANCH_INNER = """                {{- '<|im_start|>' + message.role + '\\n' + content }}"""
_PLAIN_BRANCH_INNER_PATCHED = (
    """                {{- '<|im_start|>' + message.role + '\\n' }}"""
    """{% generation %}{{- content }}{% endgeneration %}"""
)

_PLAIN_BRANCH_OUTER = """            {{- '<|im_start|>' + message.role + '\\n' + content }}"""
_PLAIN_BRANCH_OUTER_PATCHED = (
    """            {{- '<|im_start|>' + message.role + '\\n' }}"""
    """{% generation %}{{- content }}{% endgeneration %}"""
)

_IM_END = """        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}"""
_IM_END_PATCHED = """        {%- endif %}
        {% generation %}{{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}"""

_REPLACEMENTS = [
    (_THINK_BRANCH, _THINK_BRANCH_PATCHED),
    (_PLAIN_BRANCH_INNER, _PLAIN_BRANCH_INNER_PATCHED),
    (_PLAIN_BRANCH_OUTER, _PLAIN_BRANCH_OUTER_PATCHED),
    (_IM_END, _IM_END_PATCHED),
]

_PROBE_MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Question: how many nodes?"},
    {"role": "assistant", "content": "r = search_within('d1','nodes')\nprint(r)"},
    {"role": "user", "content": "TOOL OUTPUT that must not be trained on"},
    {"role": "assistant", "content": 'SUBMIT: 645 CITATIONS: ["d1"]'},
]


def patch_tokenizer_for_assistant_masking(tokenizer) -> None:
    """Install a generation-marked chat template on `tokenizer`, in place."""
    original = tokenizer.chat_template
    if original is None:
        raise ValueError("Tokenizer has no chat_template to patch")

    patched = original
    for old, new in _REPLACEMENTS:
        count = patched.count(old)
        if count != 1:
            raise ValueError(
                "Qwen3 chat template no longer matches the expected structure "
                f"(found {count} occurrences of a fragment, expected 1). "
                "Assistant-loss masking cannot be verified — refusing to train."
            )
        patched = patched.replace(old, new)

    before = tokenizer.apply_chat_template(
        _PROBE_MESSAGES, tokenize=False, enable_thinking=False
    )
    tokenizer.chat_template = patched
    after = tokenizer.apply_chat_template(
        _PROBE_MESSAGES, tokenize=False, enable_thinking=False
    )
    if before != after:
        tokenizer.chat_template = original
        raise ValueError(
            "Patched chat template changed the rendered text — training would no "
            "longer match inference. Refusing to train."
        )

    probe = tokenizer.apply_chat_template(
        _PROBE_MESSAGES,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
        enable_thinking=False,
    )
    masked = sum(probe["assistant_masks"])
    if masked == 0 or masked == len(probe["input_ids"]):
        tokenizer.chat_template = original
        raise ValueError(
            f"Assistant mask is degenerate ({masked}/{len(probe['input_ids'])} tokens). "
            "Refusing to train."
        )
