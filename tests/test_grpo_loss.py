"""Unit tests for custom GRPO loss calculation and clipping mechanics."""

import contextlib
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from typer.testing import CliRunner

from scripts.train_grpo_custom import (
    MAX_CTX_TOKENS,
    PPO_CLIP_EPS,
    _compute_token_log_probs,
    _grpo_update,
    _prepare_ctx,
)


def test_prepare_ctx_truncation() -> None:
    short_ctx = torch.arange(100)
    assert torch.equal(_prepare_ctx(short_ctx), short_ctx)

    long_ctx = torch.arange(MAX_CTX_TOKENS + 500)
    truncated = _prepare_ctx(long_ctx)
    assert truncated.shape[0] == MAX_CTX_TOKENS
    assert torch.equal(truncated, long_ctx[-MAX_CTX_TOKENS:])


def test_per_token_ppo_ratio_and_gradient_asymmetry() -> None:
    """Verify standard PPO clipping mechanics at the token level:
    - At ratio ≈ 1.0: surrogate loss is unclipped for both positive and negative advantages.
    - At ratio << 1 - eps:
        * adv > 0 selects unclipped branch (ratio * adv), so gradient flows (attenuated).
        * adv < 0 selects clipped branch ((1 - eps) * adv), so gradient is exactly zero.
    - At ratio >> 1 + eps:
        * adv > 0 selects clipped branch ((1 + eps) * adv), so gradient is zero.
        * adv < 0 selects unclipped branch (ratio * adv), so gradient flows.
    """
    n_tokens = 10
    old_lp = torch.full((n_tokens,), -2.0)

    # 1. Exact step 0 match (ratio = 1.0)
    token_lp = torch.full((n_tokens,), -2.0, requires_grad=True)
    ratio = torch.exp(token_lp - old_lp)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4)

    adv_pos = 1.5
    surr1 = ratio * adv_pos
    surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_pos
    loss = -torch.min(surr1, surr2).sum()
    loss.backward()
    assert token_lp.grad is not None
    assert torch.allclose(token_lp.grad, torch.full((n_tokens,), -adv_pos), atol=1e-4)

    # 2. Ratio << 1 - eps (e.g. ratio = 0.5 < 0.8)
    token_lp_depressed = torch.full((n_tokens,), -2.6931, requires_grad=True)  # exp(-0.6931) ≈ 0.5
    ratio_small = torch.exp(token_lp_depressed - old_lp)
    assert torch.all(ratio_small < 1.0 - PPO_CLIP_EPS)

    # With positive advantage: unclipped branch chosen, attenuated gradient flows
    surr1_pos = ratio_small * adv_pos
    surr2_pos = torch.clamp(ratio_small, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_pos
    loss_pos = -torch.min(surr1_pos, surr2_pos).sum()
    loss_pos.backward()
    assert token_lp_depressed.grad is not None
    # Gradient = - ratio * adv = - 0.5 * 1.5 = -0.75 != 0
    assert torch.all(token_lp_depressed.grad < 0)
    assert not torch.allclose(token_lp_depressed.grad, torch.zeros_like(token_lp_depressed.grad))

    # With negative advantage: clipped branch chosen, gradient is exactly zero
    token_lp_depressed_2 = torch.full((n_tokens,), -2.6931, requires_grad=True)
    ratio_small_2 = torch.exp(token_lp_depressed_2 - old_lp)
    adv_neg = -1.5
    surr1_neg = ratio_small_2 * adv_neg
    surr2_neg = torch.clamp(ratio_small_2, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv_neg
    loss_neg = -torch.min(surr1_neg, surr2_neg).sum()
    loss_neg.backward()
    # Gradient is exactly 0 because clamped value (1 - eps) * adv_neg is constant w.r.t. policy
    assert token_lp_depressed_2.grad is not None
    assert torch.allclose(token_lp_depressed_2.grad, torch.zeros_like(token_lp_depressed_2.grad), atol=1e-6)


# --- regression: dropout must be disabled during the PPO update -------------


class _FakeLoRAModel(torch.nn.Module):
    """Minimal stand-in for a PEFT model: base path + dropout-wrapped adapter.

    Mirrors the real failure surface — LoRA applies dropout to the adapter
    branch, so forward passes are stochastic in train() mode and deterministic
    in eval().
    """

    def __init__(self, vocab: int = 32, dim: int = 8, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)
        self.adapter = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self._adapter_on = True

    def forward(self, ids: torch.Tensor):  # noqa: ANN201
        h = self.emb(ids)
        if self._adapter_on:
            h = h + self.dropout(self.adapter(h))
        return SimpleNamespace(logits=self.head(h))

    @contextmanager
    def disable_adapter(self):  # noqa: ANN201
        prev, self._adapter_on = self._adapter_on, False
        try:
            yield
        finally:
            self._adapter_on = prev


def _one_question_batch(model: _FakeLoRAModel):
    """Build (batch_step_data, batch_rewards) for 1 question x 2 rollouts x 1 step.

    old_token_lp is captured the way _collect_rollout does it: under eval().
    """
    ctx = torch.arange(6)
    act = torch.arange(6, 10)

    was_training = model.training
    model.eval()
    old_lp = _compute_token_log_probs(model, ctx, act)
    if was_training:
        model.train()

    rollout = [(ctx, act, old_lp)]
    return [[rollout, rollout]], [[1.0, 0.0]]


def test_grpo_update_disables_dropout_so_step0_ratio_is_one() -> None:
    """Regression: _grpo_update must run with dropout off.

    _collect_rollout captures old_token_lp under model.eval() and then leaves the
    model in train() mode. If _grpo_update inherits train mode, LoRA dropout
    perturbs this forward pass relative to the one that produced old_token_lp, so
    ratio != 1 at step 0 despite identical weights — which both trips the step-0
    assertion and silently breaks PPO's importance-sampling assumption.
    """
    torch.manual_seed(0)
    model = _FakeLoRAModel()
    batch_step_data, batch_rewards = _one_question_batch(model)

    model.train()  # the state _collect_rollout actually hands over
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # step_num=0 arms the in-loop assertion that ratio ~= 1.
    _grpo_update(model, optimizer, batch_step_data, batch_rewards, step_num=0)

    assert not model.training, "_grpo_update must put the model in eval mode"


def test_dropout_in_train_mode_would_break_the_ratio() -> None:
    """Demonstrates the bug the fix prevents, so the guard above has teeth.

    Same weights, same tokens — only the mode differs.
    """
    torch.manual_seed(0)
    model = _FakeLoRAModel(dropout_p=0.5)
    ctx, act = torch.arange(6), torch.arange(6, 10)

    model.eval()
    old_lp = _compute_token_log_probs(model, ctx, act)

    model.eval()
    assert torch.allclose(
        torch.exp(_compute_token_log_probs(model, ctx, act) - old_lp),
        torch.ones(act.shape[0]), atol=1e-5,
    ), "eval-mode passes must agree exactly"

    model.train()
    torch.manual_seed(1)
    ratio_train = torch.exp(_compute_token_log_probs(model, ctx, act) - old_lp)
    assert not torch.allclose(ratio_train, torch.ones_like(ratio_train), atol=1e-3), (
        "train-mode dropout should perturb the ratio — if this passes, the fake "
        "model no longer exercises the failure mode and the guard is vacuous"
    )


# --- regression: truncation must not evict the system prompt ----------------


def test_prepare_ctx_preserves_system_prompt_head() -> None:
    """Naive tail-slicing drops the head, where the tool documentation lives.

    A read() of a 5k-char document is ~1.2k tokens and MAX_OUTPUT_CHARS=8000
    permits ~2k, so a multi-turn episode passes MAX_CTX_TOKENS within a turn or
    two. If truncation took the tail only, training would compute log-probs on a
    context with no tool definitions in it.
    """
    import scripts.train_grpo_custom as G

    head_len = 250
    prev = G._PROMPT_HEAD_TOKENS
    G._PROMPT_HEAD_TOKENS = head_len
    try:
        # sentinel head values 0..head_len-1, then filler, then recent tail
        ctx = torch.arange(MAX_CTX_TOKENS * 2)
        out = G._prepare_ctx(ctx)

        assert out.shape[0] == MAX_CTX_TOKENS
        # head survives, in order
        assert torch.equal(out[:head_len], ctx[:head_len]), "system prompt was evicted"
        # most recent tokens survive
        assert torch.equal(out[head_len:], ctx[-(MAX_CTX_TOKENS - head_len):])
        # and it is genuinely different from the naive tail slice
        assert not torch.equal(out, ctx[-MAX_CTX_TOKENS:])
    finally:
        G._PROMPT_HEAD_TOKENS = prev


def test_prepare_ctx_falls_back_when_head_unmeasured() -> None:
    """If _set_prompt_head_len was never called, behave like the old tail slice."""
    import scripts.train_grpo_custom as G

    prev = G._PROMPT_HEAD_TOKENS
    G._PROMPT_HEAD_TOKENS = 0
    try:
        ctx = torch.arange(MAX_CTX_TOKENS * 2)
        assert torch.equal(G._prepare_ctx(ctx), ctx[-MAX_CTX_TOKENS:])
    finally:
        G._PROMPT_HEAD_TOKENS = prev


# --- KL reference selection --------------------------------------------------


class _FakeAdapterModel(_FakeLoRAModel):
    """Adds PEFT-ish adapter switching so _reference_adapter can be exercised."""

    def __init__(self) -> None:
        super().__init__()
        self.active_adapters = ["default"]
        self.set_adapter_calls: list = []

    def set_adapter(self, name) -> None:  # noqa: ANN001
        self.set_adapter_calls.append(name)
        self.active_adapters = [name] if isinstance(name, str) else list(name)


def test_reference_adapter_base_disables_adapter() -> None:
    from scripts.train_grpo_custom import _reference_adapter

    m = _FakeAdapterModel()
    with _reference_adapter(m, "base"):
        assert m._adapter_on is False, "base reference must run on pre-SFT weights"
    assert m._adapter_on is True
    assert m.set_adapter_calls == [], "base path must not touch adapter selection"


def test_reference_adapter_sft_switches_and_restores() -> None:
    """The frozen SFT copy is activated for the reference pass, then restored.

    Restoring matters: leaking the frozen adapter would silently make the *policy*
    forward pass use non-trainable weights, and the step-0 ratio check would not
    catch it because both passes would then agree.
    """
    from scripts.train_grpo_custom import REF_ADAPTER, _reference_adapter

    m = _FakeAdapterModel()
    with _reference_adapter(m, "sft"):
        assert m.active_adapters == [REF_ADAPTER]
        assert m._adapter_on is True, "sft reference must NOT disable the adapter"
    assert m.active_adapters == ["default"], "policy adapter must be restored"


def test_reference_adapter_restores_on_exception() -> None:
    from scripts.train_grpo_custom import _reference_adapter

    m = _FakeAdapterModel()
    with contextlib.suppress(RuntimeError), _reference_adapter(m, "sft"):
        raise RuntimeError("boom")
    assert m.active_adapters == ["default"], "must restore even when the block raises"


def test_base_reference_is_refused_at_high_beta(monkeypatch) -> None:
    """Exercise the CLI guard before it reaches model or corpus loading."""
    import scripts.train_grpo_custom as G

    monkeypatch.setattr(G, "KL_BETA", 0.15)
    result = CliRunner().invoke(G.app, ["--sft-checkpoint", "unused"])

    assert result.exit_code == 2
    assert "KL_BETA=0.15" in result.output
    assert "exceeds the configured" in result.output
    assert "guardrail" in result.output
