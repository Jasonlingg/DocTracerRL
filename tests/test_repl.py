"""Tests for the persistent REPL."""

import json
import subprocess
import sys

import pytest

from src.env.repl import (
    DockerREPL,
    LocalREPL,
    PersistentREPL,
    _auto_print_trailing_expression,
)


@pytest.fixture
def repl() -> LocalREPL:
    """Create a local REPL for testing."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    yield r
    r.kill_session()


def test_state_persistence(repl: LocalREPL) -> None:
    """Variables from step 1 should be available in step 2."""
    repl.execute("x = 42")
    output = repl.execute("print(x)")
    assert "42" in output


def test_function_persistence(repl: LocalREPL) -> None:
    """Functions defined in step 1 should be callable in step 2."""
    repl.execute("def double(n): return n * 2")
    output = repl.execute("print(double(21))")
    assert "42" in output


def test_tools_available(repl: LocalREPL) -> None:
    """Tool preamble should make search/read available."""
    output = repl.execute("print(type(search))")
    assert "function" in output


def test_search_tool(repl: LocalREPL) -> None:
    """search() should return results from the corpus."""
    output = repl.execute('results = search("revenue"); print(len(results))')
    # Should have at least 1 result
    assert any(c.isdigit() and int(c) > 0 for c in output.split() if c.isdigit())


def test_search_finds_a_match_in_a_single_document_corpus(tmp_path) -> None:
    """A personal vault may begin with one note; matching terms must stay searchable."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only_note.json").write_text(json.dumps({
        "doc_id": "only_note",
        "title": "Only note",
        "text": "trajectory distillation improves a compact research agent",
    }))
    repl = LocalREPL(corpus_path=str(corpus))
    repl.start_session()
    try:
        output = repl.execute('search("trajectory")')
    finally:
        repl.kill_session()
    assert "only_note" in output


def test_read_tool(repl: LocalREPL) -> None:
    """read() should return document text."""
    output = repl.execute('text = read("apex_corp_2024_financial"); print("Apex" in text)')
    assert "True" in output


def test_bare_search_call_auto_prints_its_result(repl: LocalREPL) -> None:
    """A bare search(...) with no print() must still produce visible output —
    otherwise the model gets zero feedback and keeps retrying blind."""
    output = repl.execute('search("revenue")')
    assert output.strip()
    assert "doc_id" in output or "[" in output


def test_multiline_step_still_auto_prints_only_the_trailing_expression(
    repl: LocalREPL,
) -> None:
    output = repl.execute('x = 1\nsearch("revenue")')
    assert output.strip()


def test_print_already_present_is_not_double_wrapped() -> None:
    code = 'print(search("revenue"))'
    assert _auto_print_trailing_expression(code) == code


def test_trailing_assignment_is_left_untouched() -> None:
    code = 'results = search("revenue")'
    assert _auto_print_trailing_expression(code) == code


def test_trailing_for_loop_is_left_untouched() -> None:
    code = 'for r in search("revenue"):\n    pass'
    assert _auto_print_trailing_expression(code) == code


def test_invalid_syntax_is_returned_unchanged() -> None:
    code = "def broken("
    assert _auto_print_trailing_expression(code) == code


def test_semicolon_separated_statements_on_one_line_are_all_preserved() -> None:
    """A line-based rewrite would clobber the import when it shares a line
    with the trailing expression; the fix must not drop it."""
    rewritten = _auto_print_trailing_expression("import time; time.sleep(0)")
    assert "import time" in rewritten
    compile(rewritten, "<test>", "exec")


def test_timeout(repl: LocalREPL) -> None:
    """Long-running code should timeout."""
    output = repl.execute("import time; time.sleep(10)", timeout=2)
    assert "timeout" in output.lower() or "ERROR" in output


def test_session_cleanup() -> None:
    """kill_session should clean up without errors."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    r.execute("x = 1")
    r.kill_session()
    # Should not raise
    r.kill_session()


def test_auto_select_local() -> None:
    """PersistentREPL should fall back to local when Docker unavailable."""
    repl = PersistentREPL(use_docker=False, corpus_path="data/corpus")
    repl.start_session()
    output = repl.execute("print(1 + 1)")
    assert "2" in output
    repl.kill_session()


def test_large_previous_output_does_not_hide_current_step(repl: LocalREPL) -> None:
    repl.execute('print("x" * 9000)')
    assert repl.execute('print("CURRENT_STEP_SENTINEL")').strip() == "CURRENT_STEP_SENTINEL"


@pytest.mark.parametrize("code", [
    'raise ValueError("old failure")',
    '1 / 0',
    'if broken syntax',
    'import sys; sys.exit(3)',
])
def test_failed_step_rolls_back_and_next_action_runs(repl: LocalREPL, code: str) -> None:
    repl.execute("x = 42")
    repl.execute(code)
    assert repl.execute('print("RECOVERED", x)').strip() == "RECOVERED 42"


def test_timeout_does_not_poison_next_action(repl: LocalREPL) -> None:
    repl.execute("import time; time.sleep(10)", timeout=1)
    assert repl.execute('print("RECOVERED")', timeout=2).strip() == "RECOVERED"


def test_old_stderr_is_not_replayed_and_printed_errors_are_not_failures(repl: LocalREPL) -> None:
    repl.execute('import sys; print("old warning", file=sys.stderr); x = 7')
    assert repl.execute('print(x)').strip() == "7"
    repl.execute('print("SyntaxError"); x = 9')
    assert repl.execute('print(x)').strip() == "9"


def test_docker_transport_preserves_output_and_recovers(monkeypatch):
    """Exercise Docker's script transport/output logic with a local process.

    This is not a real Docker isolation or resource-limit test.
    """
    original_run = subprocess.run
    script = ""

    def docker_run(args, **kwargs):
        nonlocal script
        if args[1] == "create":
            return subprocess.CompletedProcess(args, 0, "test-container", "")
        if args[1] in {"start", "rm"}:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[-1] == "cat > /tmp/step.py":
            script = kwargs["input"]
            return subprocess.CompletedProcess(args, 0, "", "")
        assert "timeout" in args and "--signal=KILL" in args
        return original_run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", docker_run)
    repl = DockerREPL()
    repl.start_session()
    try:
        repl.execute('print("x" * 9000); x = 7')
        assert repl.execute('print(x)').strip() == "7"
        repl.execute('raise ValueError("bad")')
        assert repl.execute('print(x)').strip() == "7"
        # A literal heredoc delimiter is data, not a shell command boundary.
        assert "SCRIPT_EOF" in repl.execute('print("""\nSCRIPT_EOF\n""")')
    finally:
        repl.kill_session()
