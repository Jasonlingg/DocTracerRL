from src.policies.teacher_hint_policy import TeacherHintPolicy


class _RecordingPolicy:
    def __init__(self) -> None:
        self.seen_observations: list[str] = []
        self.reset_calls = 0

    def act(self, observation: str) -> str:
        self.seen_observations.append(observation)
        return "search('x')"

    def reset(self) -> None:
        self.reset_calls += 1


def test_hint_is_prepended_only_on_the_first_turn():
    inner = _RecordingPolicy()
    policy = TeacherHintPolicy(inner, hint="GROUND TRUTH: unanswerable")

    policy.act("Question: does this exist?")
    policy.act("some later observation")

    assert inner.seen_observations[0].startswith("GROUND TRUTH: unanswerable")
    assert "Question: does this exist?" in inner.seen_observations[0]
    assert inner.seen_observations[1] == "some later observation"


def test_the_returned_action_is_unmodified():
    inner = _RecordingPolicy()
    policy = TeacherHintPolicy(inner, hint="GROUND TRUTH: unanswerable")

    action = policy.act("Question: does this exist?")

    assert action == "search('x')"


def test_reset_clears_hint_state_and_forwards_to_inner():
    inner = _RecordingPolicy()
    policy = TeacherHintPolicy(inner, hint="GROUND TRUTH: unanswerable")
    policy.act("first episode question")

    policy.reset()
    policy.act("second episode question")

    assert inner.reset_calls == 1
    assert inner.seen_observations[1].startswith("GROUND TRUTH: unanswerable")


def test_works_with_a_policy_that_has_no_reset_method():
    class _NoReset:
        def act(self, observation: str) -> str:
            return "ok"

    policy = TeacherHintPolicy(_NoReset(), hint="hint")
    policy.reset()  # must not raise
    assert policy.act("obs") == "ok"
