import pytest

from scripts.train_sft import _expand_per_action, _tokenize_per_action


MESSAGES = [
    {"role": "system", "content": "Use tools."},
    {"role": "user", "content": "Question: where?"},
    {"role": "assistant", "content": "print(search('where'))"},
    {"role": "user", "content": "Found d1"},
    {"role": "assistant", "content": 'SUBMIT: There CITATIONS: ["d1"]'},
]


def test_expand_per_action_preserves_history_and_one_target():
    examples = _expand_per_action([{"messages": MESSAGES}])

    assert len(examples) == 2
    assert examples[0] == {
        "prompt": MESSAGES[:2],
        "completion": [MESSAGES[2]],
    }
    assert examples[1] == {
        "prompt": MESSAGES[:4],
        "completion": [MESSAGES[4]],
    }


class _PrefixTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt=False, enable_thinking=False
    ):
        assert tokenize is True
        assert enable_thinking is False
        ids = []
        for message in messages:
            ids += [len(message["role"]), len(message["content"])]
        if add_generation_prompt:
            ids += [999, 1000]
        elif messages[-1]["role"] == "assistant":
            ids[-2:-2] = [999, 1000]
        return ids


class _MismatchedTokenizer(_PrefixTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        if not kwargs.get("add_generation_prompt") and messages[-1]["role"] == "assistant":
            ids[len(ids) - 4] = 998
        return ids


def test_tokenize_per_action_masks_prompt_and_counts_target():
    examples = _expand_per_action([{"messages": MESSAGES}])
    rows, stats = _tokenize_per_action(_PrefixTokenizer(), examples, 100, "train")

    assert len(rows) == 2
    for row in rows:
        first_target = next(i for i, label in enumerate(row["labels"]) if label != -100)
        assert all(label == -100 for label in row["labels"][:first_target])
        assert row["labels"][first_target:] == row["input_ids"][first_target:]
    assert stats == {
        "actions": 2,
        "input_tokens": 20,
        "supervised_tokens": 4,
        "max_tokens": 12,
    }


def test_tokenize_per_action_refuses_prefix_mismatch():
    examples = _expand_per_action([{"messages": MESSAGES}])
    with pytest.raises(ValueError, match="do not start with the inference generation prefix"):
        _tokenize_per_action(_MismatchedTokenizer(), examples, 100, "train")


def test_tokenize_per_action_refuses_truncation():
    examples = _expand_per_action([{"messages": MESSAGES}])
    with pytest.raises(ValueError, match="refusing silent target truncation"):
        _tokenize_per_action(_PrefixTokenizer(), examples, 5, "train")
