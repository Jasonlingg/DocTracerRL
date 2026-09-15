"""Line-delimited JSON worker for a persistent Python execution namespace."""

from __future__ import annotations

import contextlib
import io
import json
import signal
import sys
import traceback
from typing import Any


class ExecutionTimeoutError(Exception):
    """Raised inside the worker when an action exceeds its wall-clock limit."""


def _timeout_handler(signum: int, frame: object) -> None:
    del signum, frame
    raise ExecutionTimeoutError


def execute(namespace: dict[str, Any], code: str, timeout: int) -> dict[str, Any]:
    """Execute one action, preserving the namespace for the next request."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    ok = True
    timed_out = False
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if hasattr(signal, "setitimer"):
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                exec(compile(code, "<agent-action>", "exec"), namespace)
            finally:
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
    except ExecutionTimeoutError:
        ok = False
        timed_out = True
        stderr.write(f"ERROR: Execution timed out after {timeout}s\n")
    except BaseException:
        ok = False
        traceback.print_exc(file=stderr)
    return {
        "ok": ok,
        "timed_out": timed_out,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }


def main() -> None:
    """Read requests from stdin and emit exactly one JSON response per line."""
    namespace: dict[str, Any] = {"__name__": "__agent_repl__"}
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = execute(
                namespace,
                str(request.get("code", "")),
                int(request.get("timeout", 30)),
            )
        except BaseException:
            response = {
                "ok": False,
                "timed_out": False,
                "stdout": "",
                "stderr": traceback.format_exc(),
            }
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
