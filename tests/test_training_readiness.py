"""Regressions for rollout conditioning, reward contamination and eval identity.

Uses a randomly initialized tiny transformer and local REPL; no model downloads.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from typer.testing import CliRunner

import scripts.train_grpo_custom as grpo
from scripts.run_eval import save_transcripts
from scripts.summarize_eval import load, paired_delta
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import outcome_reward
from src.eval.harness import _run_one_question, run_single


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
def test_rollout_scores_the_exact_sampled_context_and_distribution(monkeypatch, temperature):
    monkeypatch.setattr(grpo, "MAX_CTX_TOKENS", 24)
    monkeypatch.setattr(grpo, "NORM_TOKENS", 2)
    torch.manual_seed(17)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=32, n_embd=8, n_layer=1, n_head=2,
        bos_token_id=30, eos_token_id=31, pad_token_id=31,
    ))
    monkeypatch.setattr(model, "disable_adapter", nullcontext, raising=False)
    generated = []
    original_generate = model.generate

    def generate(ids, **kwargs):
        cfg = kwargs["generation_config"]
        assert cfg.top_k == 0 and cfg.top_p == 1.0
        result = original_generate(ids, **kwargs, output_scores=True)
        acts = result.sequences[0, ids.shape[1]:]
        sampled_lp = torch.stack([
            torch.log_softmax(scores[0].float(), dim=-1)[token]
            for scores, token in zip(result.scores, acts)
        ])
        generated.append((ids[0].clone(), sampled_lp))
        return result

    monkeypatch.setattr(model, "generate", generate)

    class Tokenizer:
        eos_token_id = 31

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # First six tokens represent the system and question prefix.
            length = 6 if len(messages) == 2 else 100
            return [i % 29 for i in range(length + int(add_generation_prompt))]

        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([prompt]))

        def decode(self, ids, **kwargs):
            return 'print("explore")'

    env = SimpleNamespace(reset=lambda **kw: "question",
                          step=lambda action: ("long evidence", 0.0, False, {}))
    steps, reward = grpo._collect_rollout(
        model, Tokenizer(), env, [{"question": "question"}], 0,
        max_steps=2, temperature=temperature,
    )
    assert reward == 0.0 and len(steps) == 2
    assert len(steps[1][0]) == 24
    assert torch.equal(steps[1][0][:6], torch.arange(6))
    assert torch.equal(steps[1][0][6:], torch.tensor([i % 29 for i in range(83, 101)]))
    model.eval()
    for (ctx, actions, old_lp), (sampled_ctx, sampled_lp) in zip(steps, generated):
        assert torch.equal(ctx, sampled_ctx)
        assert torch.allclose(old_lp, sampled_lp, atol=1e-5)
        new_lp, kl, _ = grpo._step_log_prob_and_kl(model, ctx, actions, temperature)
        assert torch.allclose(torch.exp(new_lp - old_lp), torch.ones_like(old_lp), atol=1e-5)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)
        (-new_lp.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_scoring_rejects_unbounded_legacy_context():
    with pytest.raises(ValueError, match="generation time"):
        grpo._scoring_context(torch.arange(grpo.MAX_CTX_TOKENS + 1))


def test_warning_and_repeated_printed_answer_never_earn_reward():
    questions = [{"id": "yes", "question": "Is this true?", "answer": "yes"}]
    env = DocumentExplorationEnv(None, questions, max_steps=6, use_docker=False)
    try:
        env.reset(0)
        obs, reward, _, _ = env.step('print("nothing")')
        assert "If yes" in obs and reward == 0.0
        for _ in range(3):
            _, reward, _, _ = env.step('print("yes")')
            assert reward == 0.0
        _, reward, done, _ = env.step("SUBMIT: no CITATIONS: []")
        assert done and reward == pytest.approx(0.2)  # empty citation sets match
    finally:
        env.close()


def test_eval_artifacts_keep_checkpoint_identity_and_recorded_outcome(tmp_path):
    paths = []
    for label, checkpoint, answer in [("3_grpo50", "adapter50", "yes"),
                                      ("4_grpo", "adapter75", "no")]:
        questions = [{"id": "q", "question": "Question", "answer": "yes"}]
        env = DocumentExplorationEnv(None, questions, use_docker=False)
        try:
            result = run_single(env, SimpleNamespace(act=lambda obs: f"SUBMIT: {answer}"), 0)
        finally:
            env.close()
        result.policy_name = "grpo_policy"
        path = save_transcripts([result], tmp_path / f"{label}.json", label,
                                {"checkpoint_id": checkpoint, "comparison_id": "same"})
        row = json.loads(path.read_text())[0]
        assert row["episode_return"] == row["outcome_reward"] == row["reward"]
        assert row["shaping_reward"] == 0
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        assert manifest["checkpoint_id"] == checkpoint
        assert manifest["run_id"] == row["run_id"]
        paths.append(path)
    groups = load(paths)
    assert len(groups) == 2
    a, b = list(groups.values())
    assert paired_delta(a, b) == pytest.approx(0.8)
    b[0]["comparison_id"] = "different"
    with pytest.raises(ValueError, match="must match"):
        paired_delta(a, b)
    with pytest.raises(FileExistsError):
        save_transcripts([result], paths[0])


def test_historical_files_are_not_merged_or_rescored(tmp_path):
    paths = [tmp_path / "old50.json", tmp_path / "old75.json"]
    row = {"policy": "grpo_policy", "question_id": "q", "reward": 0.6,
           "efficiency_bonus": 0.1}
    for path in paths:
        path.write_text(json.dumps([row]))
    assert len(load(paths)) == 2
    assert outcome_reward(row) == pytest.approx(0.5)
    assert outcome_reward({**row, "outcome_reward": 0.3}) == 0.3


def test_eval_execution_failure_is_not_a_normal_zero_score():
    def failed_policy():
        raise RuntimeError("load failed")

    result = _run_one_question(None, [{"id": "q", "question": "Question", "answer": "a"}],
                               0, "grpo_policy", failed_policy, 2, False, "data/corpus")
    assert result.status == "error"
    assert "load failed" in result.error


@pytest.mark.parametrize("group", [[1.0], [1.0, float("nan")]])
def test_invalid_grpo_reward_groups_are_rejected(group):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="finite rewards"):
        grpo._grpo_update(model, optimizer, [[[object()]] * len(group)], [group])


def test_eval_cli_saves_failure_and_exits_nonzero(tmp_path, monkeypatch):
    import scripts.run_eval as cli

    question_file = tmp_path / "questions.json"
    question_file.write_text(json.dumps([{"id": "q", "question": "Q", "answer": "A"}]))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setattr(cli, "Corpus", lambda **kw: SimpleNamespace(load=lambda: None))

    def failed_action(obs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(cli, "build_policies", lambda *args, **kw: {
        "grpo_policy": SimpleNamespace(act=failed_action),
    })
    output = tmp_path / "failed.json"
    result = CliRunner().invoke(cli.app, [
        "--questions", str(question_file), "--corpus", str(corpus),
        "--policy", "grpo_policy", "--run-label", "grpo50", "--output", str(output),
    ])
    assert result.exit_code == 1, result.output
    row = json.loads(output.read_text())[0]
    assert row["status"] == "error" and "inference failed" in row["error"]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["question_ids"] == ["q"] and manifest["seed"] == 42
    assert manifest["comparison_id"] == row["comparison_id"]
