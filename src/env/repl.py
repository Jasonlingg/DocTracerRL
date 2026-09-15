"""Persistent REPL for sandboxed code execution across episode steps.

Supports Docker containers (primary) and local subprocess fallback.
State persists via a cumulative script that grows each step.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod

from loguru import logger

from src.env.tools import TOOL_PREAMBLE

MAX_OUTPUT_CHARS = 8000
STEP_MARKER = "___STEP_OUTPUT_MARKER___"


def _auto_print_trailing_expression(code: str) -> str:
    """Rewrite a trailing bare expression statement into a print() call.

    Each step runs as `python3 script.py`, not an interactive REPL, so a bare
    `search(...)` at the end of a step produces no output at all — the model
    silently loses its own tool results and keeps retrying blind. This mirrors
    what an interactive session already does for the last expression.

    Rebuilds via ast.unparse rather than slicing source lines: statements can
    share one physical line (e.g. `import time; time.sleep(10)`), where a
    line-based rewrite would clobber everything else on that line.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if (isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "print"):
        return code
    tree.body[-1] = ast.Expr(value=ast.Call(
        func=ast.Name(id="print", ctx=ast.Load()), args=[last.value], keywords=[],
    ))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _extract_step_output(raw_output: str) -> str:
    """Extract only the latest step's output by finding the last marker."""
    if STEP_MARKER in raw_output:
        # Take everything after the last marker
        parts = raw_output.rsplit(STEP_MARKER, 1)
        output = parts[-1].lstrip("\n")
    else:
        output = raw_output

    # Truncate long output to prevent observation flooding
    if len(output) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        output = (
            output[:half]
            + f"\n\n... [{len(output) - MAX_OUTPUT_CHARS} chars truncated] ...\n\n"
            + output[-half:]
        )
    return output


def _process_output(result: subprocess.CompletedProcess) -> str:
    # stdout and stderr are captured separately; remove replayed output from
    # each stream BEFORE combining them or imposing an observation limit.
    stdout = result.stdout.rsplit(STEP_MARKER, 1)[-1].lstrip("\n")
    stderr = result.stderr.rsplit(STEP_MARKER, 1)[-1].lstrip("\n")
    return stdout + ("\nSTDERR:\n" + stderr if stderr else "")


def _step_marker() -> str:
    return (
        f'\nprint("{STEP_MARKER}", flush=True)\n'
        f'__import__("sys").stderr.write("{STEP_MARKER}\\n")\n'
        '__import__("sys").stderr.flush()\n'
    )


class BaseREPL(ABC):
    """Abstract interface for a persistent REPL session."""

    @abstractmethod
    def start_session(self) -> None: ...

    @abstractmethod
    def execute(self, code: str, timeout: int = 30) -> str: ...

    @abstractmethod
    def kill_session(self) -> None: ...


class DockerREPL(BaseREPL):
    """Persistent REPL using a Docker container with cumulative script."""

    def __init__(
        self,
        image: str = "rlm-sandbox",
        memory_limit: str = "512m",
        corpus_path: str = "data/corpus",
        extra_preamble: str = "",
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.corpus_path = os.path.abspath(corpus_path)
        self.extra_preamble = extra_preamble
        # Mount this exact host corpus at the path exposed through CORPUS_DIR.
        self._container_corpus_dir = self._resolve_container_corpus_dir(corpus_path)
        self._container_id: str | None = None
        self._cumulative_script: str = ""
        self._step: int = 0
        self._last_execution_failed = False

    @staticmethod
    def _resolve_container_corpus_dir(corpus_path: str) -> str:
        """Map a host-side corpus path to the equivalent path inside the container."""
        corpus_path = corpus_path.replace("\\", "/")
        if "musique/corpus" in corpus_path or "musique\\corpus" in corpus_path:
            return "/workspace/data/musique/corpus"
        return "/workspace/data/corpus"

    def start_session(self) -> None:
        """Create and start a Docker container, inject tool preamble.

        The supplied host corpus is mounted read-only over any baked-in corpus.
        CORPUS_DIR tells the tool preamble which mounted corpus to use.
        """
        result = subprocess.run(
            [
                "docker", "create",
                "--memory", self.memory_limit,
                "--network", "none",
                "--mount",
                f"type=bind,src={self.corpus_path},"
                f"dst={self._container_corpus_dir},readonly",
                "-e", f"CORPUS_DIR={self._container_corpus_dir}",
                "-i", self.image,
                "python3", "-i",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker create failed: {result.stderr}")

        self._container_id = result.stdout.strip()
        subprocess.run(
            ["docker", "start", self._container_id],
            capture_output=True, text=True, timeout=10,
        )
        logger.info(f"Docker container started: {self._container_id[:12]}")

        self._cumulative_script = TOOL_PREAMBLE + "\n" + self.extra_preamble
        self._step = 0
        output = self._run_script(self._cumulative_script)
        logger.debug(f"Preamble output: {output[:200]}")

    def execute(self, code: str, timeout: int = 30) -> str:
        """Execute a candidate step; retain it only when the process succeeds."""
        if self._container_id is None:
            raise RuntimeError("No active session — call start_session() first")

        self._step += 1
        previous_script = self._cumulative_script
        # Insert marker before this step's code so we can isolate its output
        marker_line = _step_marker()
        code = _auto_print_trailing_expression(code)
        self._cumulative_script += f"\n# --- Step {self._step} ---{marker_line}{code}\n"
        output = self._run_script(self._cumulative_script, timeout=timeout)

        # Extract only the latest step's output (after the marker)
        output = _extract_step_output(output)

        if self._last_execution_failed:
            logger.warning(f"Step {self._step} failed — rolling back")
            self._cumulative_script = previous_script

        return output

    def _run_script(self, script: str, timeout: int = 30) -> str:
        """Write script to container and execute it."""
        assert self._container_id is not None

        # Write script to temp file inside container
        written = subprocess.run(
            ["docker", "exec", "-i", self._container_id,
             "sh", "-c", "cat > /tmp/step.py"],
            input=script, capture_output=True, text=True, timeout=10,
        )
        if written.returncode:
            self._last_execution_failed = True
            return f"ERROR: Failed to write step script: {written.stderr}"

        # Execute the script
        try:
            result = subprocess.run(
                ["docker", "exec", "-i", self._container_id,
                 "timeout", "--signal=KILL", str(timeout), "python3", "/tmp/step.py"],
                capture_output=True, text=True, timeout=timeout + 5,
            )
            self._last_execution_failed = result.returncode != 0
            output = _process_output(result)
            if result.returncode in (124, 137):
                output += f"\nERROR: Execution timed out after {timeout}s"
        except subprocess.TimeoutExpired:
            self._last_execution_failed = True
            output = f"ERROR: Execution timed out after {timeout}s"

        return output

    def kill_session(self) -> None:
        """Remove the Docker container."""
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, text=True, timeout=10,
            )
            logger.info(f"Docker container removed: {self._container_id[:12]}")
            self._container_id = None
            self._cumulative_script = ""
            self._step = 0


class LocalREPL(BaseREPL):
    """Persistent REPL using a local subprocess with cumulative script.

    Fallback when Docker is unavailable.
    """

    def __init__(self, corpus_path: str = "data/corpus", extra_preamble: str = "") -> None:
        self.corpus_path = os.path.abspath(corpus_path)
        self.extra_preamble = extra_preamble
        self._cumulative_script: str = ""
        self._step: int = 0
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._last_execution_failed = False

    def start_session(self) -> None:
        """Initialize the local REPL session with tool preamble."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cumulative_script = TOOL_PREAMBLE + "\n" + self.extra_preamble
        self._step = 0

        output = self._run_script(self._cumulative_script)
        logger.info("Local REPL session started")
        logger.debug(f"Preamble output: {output[:200]}")

    def execute(self, code: str, timeout: int = 30) -> str:
        """Execute a candidate step; retain it only when the process succeeds."""
        if self._tmpdir is None:
            raise RuntimeError("No active session — call start_session() first")
        self._step += 1
        previous_script = self._cumulative_script
        marker_line = _step_marker()
        code = _auto_print_trailing_expression(code)
        self._cumulative_script += f"\n# --- Step {self._step} ---{marker_line}{code}\n"
        output = self._run_script(self._cumulative_script, timeout=timeout)

        # Extract only the latest step's output (after the marker)
        output = _extract_step_output(output)

        if self._last_execution_failed:
            logger.warning(f"Step {self._step} failed — rolling back")
            self._cumulative_script = previous_script

        return output

    def _run_script(self, script: str, timeout: int = 30) -> str:
        """Write cumulative script to temp file and execute."""
        if self._tmpdir is None:
            raise RuntimeError("No active session — call start_session() first")

        script_path = os.path.join(self._tmpdir.name, "step.py")
        with open(script_path, "w") as f:
            f.write(script)

        env = os.environ.copy()
        env["CORPUS_DIR"] = self.corpus_path

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
                env=env,
            )
            self._last_execution_failed = result.returncode != 0
            output = _process_output(result)
        except subprocess.TimeoutExpired:
            self._last_execution_failed = True
            output = f"ERROR: Execution timed out after {timeout}s"

        return output

    def kill_session(self) -> None:
        """Clean up temp directory."""
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._cumulative_script = ""
            self._step = 0
            logger.info("Local REPL session ended")


class PersistentREPL:
    """Auto-selecting REPL — uses Docker if available, falls back to local."""

    def __init__(
        self,
        use_docker: bool | None = None,
        image: str = "rlm-sandbox",
        memory_limit: str = "512m",
        corpus_path: str = "data/corpus",
        extra_preamble: str = "",
    ) -> None:
        if use_docker is None:
            use_docker = self._docker_available(image)

        if use_docker:
            logger.info("Using Docker REPL")
            self._impl: BaseREPL = DockerREPL(
                image=image, memory_limit=memory_limit, corpus_path=corpus_path,
                extra_preamble=extra_preamble,
            )
        else:
            logger.info("Docker unavailable — using local REPL fallback")
            self._impl = LocalREPL(corpus_path=corpus_path, extra_preamble=extra_preamble)

    @staticmethod
    def _docker_available(image: str) -> bool:
        """Check if Docker is running and the sandbox image exists."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start_session(self) -> None:
        self._impl.start_session()

    def execute(self, code: str, timeout: int = 30) -> str:
        return self._impl.execute(code, timeout=timeout)

    def kill_session(self) -> None:
        self._impl.kill_session()
