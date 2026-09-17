"""Guards the assistant-loss masking patch for Qwen3.

Without the patch, `return_assistant_tokens_mask=True` returns an all-zero mask
and only warns — training silently becomes a no-op. These tests fail loudly if
that protection regresses.
"""

import pytest

from src.policies.qwen3_chat_template import patch_tokenizer_for_assistant_masking

MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Question: how many nodes?"},
    {"role": "assistant", "content": "r = search_within('d1','nodes')"},
    {"role": "user", "content": "TOOL OUTPUT that must not be trained on"},
    {"role": "assistant", "content": 'SUBMIT: 645 CITATIONS: ["d1"]'},
]


@pytest.fixture(scope="module")
def _downloaded_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as exc:  # offline / no hub access
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")


@pytest.fixture
def tokenizer(_downloaded_tokenizer):
    """Restores the stock template after each test, since the patch is in place."""
    original = _downloaded_tokenizer.chat_template
    yield _downloaded_tokenizer
    _downloaded_tokenizer.chat_template = original


def test_stock_template_produces_an_all_zero_mask(tokenizer):
    """Documents the bug being guarded against: unpatched Qwen3 masks nothing."""
    out = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, return_assistant_tokens_mask=True,
        return_dict=True, enable_thinking=False,
    )
    assert sum(out["assistant_masks"]) == 0


def test_patch_makes_the_mask_cover_only_assistant_turns(tokenizer):
    patch_tokenizer_for_assistant_masking(tokenizer)
    out = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, return_assistant_tokens_mask=True,
        return_dict=True, enable_thinking=False,
    )
    masked = tokenizer.decode(
        [i for i, m in zip(out["input_ids"], out["assistant_masks"]) if m]
    )
    assert "search_within" in masked
    assert "SUBMIT: 645" in masked
    assert "TOOL OUTPUT" not in masked
    assert "You are an agent" not in masked


def test_patch_keeps_rendered_text_identical(tokenizer):
    """Training format must match what BaseQwenPolicy.act() renders at inference."""
    before = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, enable_thinking=False
    )
    patch_tokenizer_for_assistant_masking(tokenizer)
    after = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, enable_thinking=False
    )
    assert before == after


def test_patch_refuses_when_template_structure_is_unrecognised(tokenizer):
    tokenizer.chat_template = "{{ 'totally different template' }}"
    with pytest.raises(ValueError, match="no longer matches the expected structure"):
        patch_tokenizer_for_assistant_masking(tokenizer)
