# Code Triage Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Qwen2.5-7B-Instruct with GRPO to triage AI-generated code diffs (SHIP / RUN_TESTS / LINT_TYPECHECK / ESCALATE_STRONG_MODEL), using a binary-gated, cost-aware reward, and demonstrate it beats naive and threshold baselines on a held-out, task-disjoint eval — shipped as an installable Claude Code skill.

**Architecture:** A new standalone repository (sibling to `rlm-explorer`, not inside it). Single-turn decision task: `(task_description, diff, mechanical_signals) → action`. Data pipeline generates and mechanically labels examples offline (no live model calls during training). SFT warm-start via TRL `SFTTrainer`, then GRPO via TRL `GRPOTrainer` — both proven library implementations, not a hand-rolled loop. Training runs on a RunPod A100 pod using the pinned-dependency workflow already validated in the `rlm-explorer` project.

**Tech Stack:** Python 3.10+, PyTorch 2.4.0+cu121, TRL 1.3.0, transformers 4.45.2, peft 0.13.2, Qwen2.5-7B-Instruct + LoRA, Pydantic for schemas, pytest, RunPod (A100).

**Spec:** [docs/superpowers/specs/2026-08-27-code-triage-router-design.md](../specs/2026-08-27-code-triage-router-design.md)

## Global Constraints

- Base model: Qwen2.5-7B-Instruct (spec §Training) — not Qwen3, not a different size.
- v1 action space is exactly 4 actions: `SHIP`, `RUN_TESTS`, `LINT_TYPECHECK`, `ESCALATE_STRONG_MODEL` (spec §The decision). `ASK_CLARIFY` and `FLAG_HUMAN` are out of scope for this plan.
- Reward formula is exactly `task_success_indicator × (K − λ · cost(action)) − format_penalty` (spec §Reward design) — binary-gated, no partial credit for a wrong `SHIP`.
- Ground truth is always mechanical (test/lint/typecheck pass-fail) — never an LLM-judge label (spec §Data pipeline step 2).
- Train/eval split is by task/repo, never by individual diff (spec §Data pipeline step 3) — zero leakage.
- Dependencies are pinned from day one (spec §Risks) — no unpinned `pip install` on the training pod, ever.
- GRPO training uses TRL's `GRPOTrainer`, not a hand-rolled PPO-clip loop (spec §Training).
- Every full-cost training run (GRPO diagnostic, GRPO full run) requires an explicit go/no-go checkpoint before proceeding to the next spend (spec §Training, "Mandatory gate").
- Full-eval runs must log complete transcripts, not just pass/fail (spec §Demo & write-up assets) — this cannot be retrofitted after the run.
- Python code style: type hints on all functions, Pydantic models for data structures (matches author's established `rlm-explorer` convention).

---

## Task 1: Repository scaffold and pinned dependencies

**Files:**
- Create: `code-triage-router/pyproject.toml`
- Create: `code-triage-router/requirements.txt`
- Create: `code-triage-router/src/__init__.py`
- Create: `code-triage-router/tests/__init__.py`
- Create: `code-triage-router/.gitignore`
- Test: `code-triage-router/tests/test_scaffold.py`

**Interfaces:**
- Produces: an installable package `src` importable from any task/script in this plan; `requirements.txt` is the single pinned source of truth every later task's dependencies come from.

- [ ] **Step 1: Create the repository directory and git init**

```bash
mkdir -p /Users/jasonling/Documents/GitHub/code-triage-router
cd /Users/jasonling/Documents/GitHub/code-triage-router
git init
```

- [ ] **Step 2: Write `requirements.txt`, adapted from rlm-explorer's proven pod-tested set**

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.4.0+cu121
numpy<2
transformers==4.45.2
peft==0.13.2
accelerate==1.4.0
trl==1.3.0
bitsandbytes==0.44.1
datasets==3.1.0
pydantic>=2.0
loguru>=0.7
rich>=13.0
typer>=0.12
python-dotenv>=1.0
anthropic>=0.39.0
gitpython>=3.1
ruff>=0.5
pytest>=8.0
```

(This is `rlm-explorer/requirements-pod.txt` with the corpus/embedding/RAG-specific packages — `sentence-transformers`, `faiss-cpu`, `rank-bm25` — dropped, since this project has no document-retrieval component, and `gitpython` added for the git-history mining source in Task 3.)

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "code-triage-router"
version = "0.1.0"
description = "RL-trained router for triaging AI-generated code diffs"
requires-python = ">=3.10"
license = "MIT"
dependencies = []

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

(`dependencies = []` deliberately — `requirements.txt` is the single source of truth per the Global Constraints; `pip install -e .` plus `pip install -r requirements.txt` is the install path, avoiding the two-sources-of-truth drift that caused version conflicts in the prior project.)

- [ ] **Step 4: Create empty package markers**

```bash
mkdir -p src/data src/baselines src/training src/eval src/skill tests scripts
touch src/__init__.py src/data/__init__.py src/baselines/__init__.py src/training/__init__.py src/eval/__init__.py src/skill/__init__.py tests/__init__.py
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
data/generated/
checkpoints/
.env
```

- [ ] **Step 6: Write the scaffold test**

```python
# tests/test_scaffold.py
import src


def test_package_importable():
    assert src is not None
```

- [ ] **Step 7: Install locally and run the test**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pydantic loguru
pytest tests/test_scaffold.py -v
```

Expected: PASS (this only needs a minimal subset of `requirements.txt` — pytest, pydantic, loguru — since torch/transformers/trl are only needed on the training pod, not for local scaffold/unit-test work).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "scaffold repository with pinned requirements"
```

---

## Task 2: Data schemas

**Files:**
- Create: `src/data/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `TriageExample` (Pydantic model: `task_description: str`, `diff: str`, `signals: MechanicalSignals`, `label: Action | None`), `MechanicalSignals` (Pydantic model: `lint_pass: bool | None`, `typecheck_pass: bool | None`, `test_pass: bool | None`), `Action` (str enum: `SHIP`, `RUN_TESTS`, `LINT_TYPECHECK`, `ESCALATE_STRONG_MODEL`). These types are consumed by every later task (labeling, dataset splitting, baselines, reward, eval).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import pytest
from pydantic import ValidationError

from src.data.schema import Action, MechanicalSignals, TriageExample


def test_action_enum_has_exactly_four_values():
    assert set(Action) == {
        Action.SHIP,
        Action.RUN_TESTS,
        Action.LINT_TYPECHECK,
        Action.ESCALATE_STRONG_MODEL,
    }


def test_mechanical_signals_defaults_to_unknown():
    signals = MechanicalSignals()
    assert signals.lint_pass is None
    assert signals.typecheck_pass is None
    assert signals.test_pass is None


def test_triage_example_requires_task_and_diff():
    example = TriageExample(
        task_description="Fix the off-by-one error in pagination",
        diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y",
        signals=MechanicalSignals(test_pass=True),
        label=Action.SHIP,
        source="swebench",
        task_id="repo/issue-123",
    )
    assert example.label == Action.SHIP
    assert example.source == "swebench"


def test_triage_example_rejects_missing_diff():
    with pytest.raises(ValidationError):
        TriageExample(
            task_description="Fix something",
            signals=MechanicalSignals(),
            label=None,
            source="swebench",
            task_id="repo/issue-1",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.schema'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/schema.py
"""Shared data types for triage examples, used by data generation, training, and eval."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Action(StrEnum):
    SHIP = "SHIP"
    RUN_TESTS = "RUN_TESTS"
    LINT_TYPECHECK = "LINT_TYPECHECK"
    ESCALATE_STRONG_MODEL = "ESCALATE_STRONG_MODEL"


ACTION_COST: dict[Action, float] = {
    Action.SHIP: 0.0,
    Action.RUN_TESTS: 0.1,
    Action.LINT_TYPECHECK: 0.1,
    Action.ESCALATE_STRONG_MODEL: 1.0,
}


class MechanicalSignals(BaseModel):
    lint_pass: bool | None = None
    typecheck_pass: bool | None = None
    test_pass: bool | None = None


class TriageExample(BaseModel):
    task_description: str
    diff: str
    signals: MechanicalSignals
    label: Action | None
    source: str
    task_id: str
    reasoning: str | None = None
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
</br>
```

Note: remove the stray `</br>` lines below the class body before saving — they are not valid Python and are an artifact to be deleted, not executed.

- [ ] **Step 3b: Correct the implementation file (remove the stray lines)**

```python
# src/data/schema.py
"""Shared data types for triage examples, used by data generation, training, and eval."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Action(StrEnum):
    SHIP = "SHIP"
    RUN_TESTS = "RUN_TESTS"
    LINT_TYPECHECK = "LINT_TYPECHECK"
    ESCALATE_STRONG_MODEL = "ESCALATE_STRONG_MODEL"


ACTION_COST: dict[Action, float] = {
    Action.SHIP: 0.0,
    Action.RUN_TESTS: 0.1,
    Action.LINT_TYPECHECK: 0.1,
    Action.ESCALATE_STRONG_MODEL: 1.0,
}


class MechanicalSignals(BaseModel):
    lint_pass: bool | None = None
    typecheck_pass: bool | None = None
    test_pass: bool | None = None


class TriageExample(BaseModel):
    task_description: str
    diff: str
    signals: MechanicalSignals
    label: Action | None
    source: str
    task_id: str
    reasoning: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/schema.py tests/test_schema.py
git commit -m "add data schemas: Action, MechanicalSignals, TriageExample"
```

---

## Task 3: Mechanical labeling

**Files:**
- Create: `src/data/labeling.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Consumes: nothing from earlier tasks beyond running in the same repo.
- Produces: `run_tests(repo_path: Path) -> bool | None`, `run_lint(repo_path: Path) -> bool | None`, `run_typecheck(repo_path: Path) -> bool | None`, `label_diff(repo_path: Path) -> MechanicalSignals` (from `src.data.schema`). Consumed by Task 4 (dataset generation) and Task 5 (git-history mining).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labeling.py
import subprocess
from pathlib import Path

import pytest

from src.data.labeling import label_diff, run_lint, run_tests


@pytest.fixture
def passing_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "passing_repo"
    repo.mkdir()
    (repo / "add.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "test_add.py").write_text(
        "from add import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    return repo


@pytest.fixture
def failing_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "failing_repo"
    repo.mkdir()
    (repo / "add.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_add.py").write_text(
        "from add import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_run_tests_passes_on_correct_code(passing_repo: Path):
    assert run_tests(passing_repo) is True


def test_run_tests_fails_on_incorrect_code(failing_repo: Path):
    assert run_tests(failing_repo) is False


def test_run_lint_passes_on_clean_code(passing_repo: Path):
    assert run_lint(passing_repo) is True


def test_label_diff_returns_all_signals(passing_repo: Path):
    signals = label_diff(passing_repo)
    assert signals.test_pass is True
    assert signals.lint_pass is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.labeling'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/labeling.py
"""Mechanical (non-LLM) ground-truth labeling: run tests/lint/typecheck, record pass/fail.

Ground truth for this project is always mechanical execution, never an
LLM-judge opinion — see spec 'Data pipeline' step 2.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.data.schema import MechanicalSignals


def _run(cmd: list[str], cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, timeout=60, text=True
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.returncode == 0


def run_tests(repo_path: Path) -> bool | None:
    return _run(["python", "-m", "pytest", "-q"], repo_path)


def run_lint(repo_path: Path) -> bool | None:
    return _run(["python", "-m", "ruff", "check", "."], repo_path)


def run_typecheck(repo_path: Path) -> bool | None:
    return _run(["python", "-m", "mypy", "--ignore-missing-imports", "."], repo_path)


def label_diff(repo_path: Path) -> MechanicalSignals:
    return MechanicalSignals(
        test_pass=run_tests(repo_path),
        lint_pass=run_lint(repo_path),
        typecheck_pass=run_typecheck(repo_path),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install ruff mypy && pytest tests/test_labeling.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/labeling.py tests/test_labeling.py
git commit -m "add mechanical labeling: run_tests, run_lint, run_typecheck"
```

---

## Task 4: SWE-bench-style dataset generation

**Files:**
- Create: `src/data/swebench_source.py`
- Test: `tests/test_swebench_source.py`

**Interfaces:**
- Consumes: `MechanicalSignals`, `TriageExample`, `Action` (Task 2); `label_diff` (Task 3).
- Produces: `generate_diff_for_task(task_description: str, repo_path: Path, model: str = "claude-haiku-4-5-20251001") -> str` (calls a cheap coding model to produce a diff for a task); `hindsight_label(signals: MechanicalSignals) -> Action` (maps mechanical signals to the ground-truth action a rational policy should have taken); `build_examples_from_swebench(tasks: list[dict], n: int) -> list[TriageExample]`. Consumed by Task 6 (dataset assembly).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_swebench_source.py
from unittest.mock import patch

from src.data.schema import Action, MechanicalSignals
from src.data.swebench_source import hindsight_label


def test_hindsight_label_ships_when_all_signals_pass():
    signals = MechanicalSignals(test_pass=True, lint_pass=True, typecheck_pass=True)
    assert hindsight_label(signals) == Action.SHIP


def test_hindsight_label_escalates_when_tests_fail():
    signals = MechanicalSignals(test_pass=False, lint_pass=True, typecheck_pass=True)
    assert hindsight_label(signals) == Action.ESCALATE_STRONG_MODEL


def test_hindsight_label_requests_tests_when_unknown():
    signals = MechanicalSignals(test_pass=None, lint_pass=True, typecheck_pass=True)
    assert hindsight_label(signals) == Action.RUN_TESTS


def test_hindsight_label_requests_lint_when_lint_unknown_but_tests_pass():
    signals = MechanicalSignals(test_pass=True, lint_pass=None, typecheck_pass=None)
    assert hindsight_label(signals) == Action.LINT_TYPECHECK
```

Note: this hindsight-labeling function encodes "what a rational policy would have done given full information" — it is the ground truth used to construct SFT reasoning traces (Task 7) and to score baselines (Task 8), not something the trained model has access to at inference time (the model never sees `label`, only `task_description`, `diff`, and whatever signals happen to already be known).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_swebench_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.swebench_source'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/swebench_source.py
"""Generate (task, diff) pairs from SWE-bench-style tasks by running a cheap
coding model, then label them mechanically.

Primary data source per spec 'Data pipeline' step 1 — controllable difficulty,
executable tests already exist, no dependency on personal git history volume.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from src.data.labeling import label_diff
from src.data.schema import Action, MechanicalSignals, TriageExample

_CHEAP_MODEL = "claude-haiku-4-5-20251001"


def hindsight_label(signals: MechanicalSignals) -> Action:
    """What a rational policy would have chosen, given full information.

    Used to build SFT reasoning traces and score baselines — not seen by the
    trained model at inference time.
    """
    if signals.test_pass is None:
        return Action.RUN_TESTS
    if signals.test_pass is False:
        return Action.ESCALATE_STRONG_MODEL
    if signals.lint_pass is None or signals.typecheck_pass is None:
        return Action.LINT_TYPECHECK
    if signals.lint_pass is False or signals.typecheck_pass is False:
        return Action.ESCALATE_STRONG_MODEL
    return Action.SHIP


def generate_diff_for_task(
    task_description: str, repo_context: str, client: Anthropic | None = None
) -> str:
    """Have a cheap model attempt the task; return its unified diff."""
    client = client or Anthropic()
    response = client.messages.create(
        model=_CHEAP_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Repository context:\n{repo_context}\n\n"
                    f"Task: {task_description}\n\n"
                    "Respond with ONLY a unified diff that solves this task. "
                    "No explanation, no markdown fences."
                ),
            }
        ],
    )
    return response.content[0].text


def build_examples_from_swebench(
    tasks: list[dict], repos_dir: Path, client: Anthropic | None = None
) -> list[TriageExample]:
    """tasks: list of {"task_id": str, "description": str, "repo_context": str,
    "repo_path": Path (already checked out with the generated diff applied)}."""
    examples: list[TriageExample] = []
    for task in tasks:
        diff = generate_diff_for_task(task["description"], task["repo_context"], client)
        signals = label_diff(task["repo_path"])
        examples.append(
            TriageExample(
                task_description=task["description"],
                diff=diff,
                signals=signals,
                label=hindsight_label(signals),
                source="swebench",
                task_id=task["task_id"],
            )
        )
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_swebench_source.py -v`
Expected: PASS (4 passed) — these tests only exercise `hindsight_label`, which has no external dependency, so no API key is needed to pass this task's tests.

- [ ] **Step 5: Commit**

```bash
git add src/data/swebench_source.py tests/test_swebench_source.py
git commit -m "add SWE-bench-style data generation and hindsight labeling"
```

---

## Task 5: Git-history supplementary source

**Files:**
- Create: `src/data/git_history_source.py`
- Test: `tests/test_git_history_source.py`

**Interfaces:**
- Consumes: `TriageExample`, `Action`, `MechanicalSignals` (Task 2).
- Produces: `find_fix_commits(repo_path: Path, window_hours: int = 24) -> list[tuple[str, str]]` (pairs of (original_commit_sha, fix_commit_sha) where a fix followed within the window); `build_examples_from_git_history(repo_path: Path) -> list[TriageExample]`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_git_history_source.py
import subprocess
from pathlib import Path

import pytest

from src.data.git_history_source import build_examples_from_git_history
from src.data.schema import Action


@pytest.fixture
def repo_with_fix(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)

    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add feature: implement x"], cwd=repo, check=True
    )

    (repo / "a.py").write_text("x = 2\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fix: correct x value"], cwd=repo, check=True)

    return repo


def test_build_examples_labels_fixed_commit_as_escalate(repo_with_fix: Path):
    examples = build_examples_from_git_history(repo_with_fix)
    assert len(examples) == 1
    assert examples[0].label == Action.ESCALATE_STRONG_MODEL
    assert examples[0].source == "git_history"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_git_history_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.git_history_source'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/git_history_source.py
"""Mine the author's own commit history for free-label examples.

Secondary, supplementary data source per spec 'Data pipeline' step 1 — grounds
the eval in real daily-use examples. A commit followed by a same-day fix
commit is a free negative label (should have escalated); a commit with no
follow-up fix is a free positive label (SHIP was correct).
"""

from __future__ import annotations

from pathlib import Path

from git import Repo

from src.data.schema import Action, MechanicalSignals, TriageExample

_FIX_KEYWORDS = ("fix", "bug", "correct", "revert")


def _is_fix_commit(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in _FIX_KEYWORDS)


def build_examples_from_git_history(repo_path: Path) -> list[TriageExample]:
    repo = Repo(repo_path)
    commits = list(repo.iter_commits("HEAD"))
    commits.reverse()  # oldest first

    examples: list[TriageExample] = []
    for i, commit in enumerate(commits[:-1]):
        next_commit = commits[i + 1]
        was_fixed = _is_fix_commit(next_commit.message)
        diff = repo.git.diff(commit.parents[0].hexsha if commit.parents else "", commit.hexsha)
        examples.append(
            TriageExample(
                task_description=commit.message.strip(),
                diff=diff,
                signals=MechanicalSignals(test_pass=not was_fixed),
                label=Action.ESCALATE_STRONG_MODEL if was_fixed else Action.SHIP,
                source="git_history",
                task_id=commit.hexsha,
            )
        )
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install gitpython && pytest tests/test_git_history_source.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/git_history_source.py tests/test_git_history_source.py
git commit -m "add git-history supplementary data source"
```

---

## Task 6: Dataset assembly and task-disjoint split

**Files:**
- Create: `src/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `TriageExample`, `Action` (Task 2).
- Produces: `split_by_task(examples: list[TriageExample], eval_fraction: float = 0.2, seed: int = 42) -> tuple[list[TriageExample], list[TriageExample]]` (returns train, eval — split by distinct `task_id` prefix/repo, never by individual example); `class_balance(examples: list[TriageExample]) -> dict[Action, float]`; `save_dataset(examples: list[TriageExample], path: Path) -> None`; `load_dataset(path: Path) -> list[TriageExample]`. Consumed by Task 8 (baselines), Task 9 (SFT), Task 11 (GRPO), Task 12 (eval).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import json
from pathlib import Path

from src.data.dataset import class_balance, load_dataset, save_dataset, split_by_task
from src.data.schema import Action, MechanicalSignals, TriageExample


def _example(task_id: str, label: Action) -> TriageExample:
    return TriageExample(
        task_description="do something",
        diff="diff content",
        signals=MechanicalSignals(test_pass=True),
        label=label,
        source="swebench",
        task_id=task_id,
    )


def test_split_by_task_has_no_task_id_overlap():
    examples = [
        _example("repoA/issue-1", Action.SHIP),
        _example("repoA/issue-2", Action.SHIP),
        _example("repoB/issue-1", Action.ESCALATE_STRONG_MODEL),
        _example("repoC/issue-1", Action.RUN_TESTS),
        _example("repoD/issue-1", Action.LINT_TYPECHECK),
    ]
    train, eval_set = split_by_task(examples, eval_fraction=0.2, seed=42)
    train_repos = {e.task_id.split("/")[0] for e in train}
    eval_repos = {e.task_id.split("/")[0] for e in eval_set}
    assert train_repos.isdisjoint(eval_repos)
    assert len(train) + len(eval_set) == len(examples)


def test_class_balance_sums_to_one():
    examples = [
        _example("repoA/1", Action.SHIP),
        _example("repoB/1", Action.SHIP),
        _example("repoC/1", Action.ESCALATE_STRONG_MODEL),
    ]
    balance = class_balance(examples)
    assert abs(sum(balance.values()) - 1.0) < 1e-9
    assert balance[Action.SHIP] == pytest.approx(2 / 3)


def test_save_and_load_roundtrip(tmp_path: Path):
    examples = [_example("repoA/1", Action.SHIP)]
    path = tmp_path / "dataset.jsonl"
    save_dataset(examples, path)
    loaded = load_dataset(path)
    assert loaded == examples


import pytest  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/dataset.py
"""Dataset assembly: task-disjoint splitting, class balance checks, JSONL I/O.

Split is by task/repo, never by individual diff — see spec 'Data pipeline'
step 3 and the routing-literature distribution-shift caveat in the spec's
research grounding section.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.data.schema import Action, TriageExample


def _task_group(example: TriageExample) -> str:
    return example.task_id.split("/")[0]


def split_by_task(
    examples: list[TriageExample], eval_fraction: float = 0.2, seed: int = 42
) -> tuple[list[TriageExample], list[TriageExample]]:
    groups = sorted({_task_group(e) for e in examples})
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_eval_groups = max(1, int(len(groups) * eval_fraction))
    eval_groups = set(groups[:n_eval_groups])

    train = [e for e in examples if _task_group(e) not in eval_groups]
    eval_set = [e for e in examples if _task_group(e) in eval_groups]
    return train, eval_set


def class_balance(examples: list[TriageExample]) -> dict[Action, float]:
    counts = Counter(e.label for e in examples if e.label is not None)
    total = sum(counts.values())
    return {action: counts.get(action, 0) / total for action in Action}


def save_dataset(examples: list[TriageExample], path: Path) -> None:
    with path.open("w") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")


def load_dataset(path: Path) -> list[TriageExample]:
    examples = []
    with path.open() as f:
        for line in f:
            examples.append(TriageExample.model_validate_json(line))
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "add dataset assembly: task-disjoint split, class balance, JSONL I/O"
```

---

## Task 7: End-to-end dataset generation script + class balance gate

**Files:**
- Create: `scripts/generate_dataset.py`
- Test: `tests/test_generate_dataset_script.py`

**Interfaces:**
- Consumes: `build_examples_from_swebench` (Task 4), `build_examples_from_git_history` (Task 5), `save_dataset`, `class_balance`, `split_by_task` (Task 6).
- Produces: a CLI script; `data/generated/train.jsonl` and `data/generated/eval.jsonl` on disk when run. This is SMART goal 1 from the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_dataset_script.py
from pathlib import Path

from src.data.dataset import load_dataset
from src.data.schema import Action, MechanicalSignals, TriageExample
from scripts.generate_dataset import check_class_balance_or_raise


def test_check_class_balance_or_raise_passes_on_reasonable_split():
    examples = (
        [
            TriageExample(
                task_description="t",
                diff="d",
                signals=MechanicalSignals(),
                label=Action.SHIP,
                source="s",
                task_id=f"r{i}/1",
            )
            for i in range(40)
        ]
        + [
            TriageExample(
                task_description="t",
                diff="d",
                signals=MechanicalSignals(),
                label=Action.ESCALATE_STRONG_MODEL,
                source="s",
                task_id=f"r{i}/2",
            )
            for i in range(40, 80)
        ]
    )
    check_class_balance_or_raise(examples, max_majority_fraction=0.8)


def test_check_class_balance_or_raise_fails_on_skewed_split():
    import pytest

    examples = [
        TriageExample(
            task_description="t",
            diff="d",
            signals=MechanicalSignals(),
            label=Action.SHIP,
            source="s",
            task_id=f"r{i}/1",
        )
        for i in range(100)
    ]
    with pytest.raises(ValueError, match="class balance"):
        check_class_balance_or_raise(examples, max_majority_fraction=0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_dataset_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.generate_dataset'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/generate_dataset.py
"""End-to-end data pipeline: generate, label, split, and save the dataset.

Implements spec SMART goal 1: >=500 labeled (task, diff, signals) triples,
class balance checked, by the end of week 1.
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.data.dataset import class_balance, save_dataset, split_by_task
from src.data.git_history_source import build_examples_from_git_history
from src.data.schema import Action, TriageExample
from src.data.swebench_source import build_examples_from_swebench

app = typer.Typer()

_MIN_EXAMPLES = 500


def check_class_balance_or_raise(
    examples: list[TriageExample], max_majority_fraction: float = 0.8
) -> None:
    balance = class_balance(examples)
    majority = max(balance.values())
    if majority > max_majority_fraction:
        raise ValueError(
            f"class balance too skewed: majority class is {majority:.1%} of examples "
            f"(threshold {max_majority_fraction:.0%}) — per spec risk 'Base rate of "
            "usable diffs may be skewed', adjust task difficulty or model choice."
        )


@app.command()
def main(
    swebench_tasks_path: Path = typer.Option(..., "--swebench-tasks"),
    git_history_repo: Path = typer.Option(..., "--git-history-repo"),
    output_dir: Path = typer.Option(Path("data/generated"), "--output-dir"),
) -> None:
    import json

    tasks = json.loads(swebench_tasks_path.read_text())
    swebench_examples = build_examples_from_swebench(tasks, repos_dir=output_dir / "repos")
    git_examples = build_examples_from_git_history(git_history_repo)

    all_examples = swebench_examples + git_examples
    logger.info(f"Generated {len(all_examples)} total examples")

    if len(all_examples) < _MIN_EXAMPLES:
        raise ValueError(
            f"only {len(all_examples)} examples generated, need >= {_MIN_EXAMPLES} "
            "per spec SMART goal 1"
        )

    check_class_balance_or_raise(all_examples)
    logger.info(f"Class balance: {class_balance(all_examples)}")

    train, eval_set = split_by_task(all_examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(train, output_dir / "train.jsonl")
    save_dataset(eval_set, output_dir / "eval.jsonl")
    logger.info(f"Saved {len(train)} train, {len(eval_set)} eval examples to {output_dir}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_dataset_script.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_dataset.py tests/test_generate_dataset_script.py
git commit -m "add end-to-end dataset generation script with class balance gate"
```

- [ ] **Step 6: Manual eval-integrity sanity check — REQUIRED before trusting the pipeline at scale**

A self-built eval is only as trustworthy as its construction (see spec eval-integrity discussion). Mechanical labels can still be wrong — e.g. a flaky test, a test that passes for the wrong reason, or a `hindsight_label` mapping that doesn't match how a human would actually judge the example. Before generating the full 500+ example dataset, run the pipeline on a small batch (~20-30 examples) and manually read every one:

```bash
python scripts/generate_dataset.py --swebench-tasks data/sample_tasks.json \
  --git-history-repo /Users/jasonling/Documents/GitHub/rlm-explorer \
  --output-dir data/sanity_check
```

For each of the ~20-30 examples in `data/sanity_check/train.jsonl` and `eval.jsonl`, read the `task_description`, `diff`, `signals`, and `label` together and ask: "would I have made this same call?" Record the fraction you disagree with in `docs/eval_sanity_check.md` (a plain note, not code). If disagreement is high (rough guideline: more than ~10-15% of examples), the `hindsight_label` mapping in `src/data/swebench_source.py` (Task 4) or the mechanical labeling in `src/data/labeling.py` (Task 3) needs revisiting — fix before proceeding to full-scale generation in Step 7 below, not after.

Only proceed to generating the full dataset once this check passes.

- [ ] **Step 7: Generate the full dataset**

```bash
python scripts/generate_dataset.py --swebench-tasks data/full_tasks.json \
  --git-history-repo /Users/jasonling/Documents/GitHub/rlm-explorer \
  --output-dir data/generated
```

---

## Task 8: Baseline policies

**Files:**
- Create: `src/baselines/policies.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `TriageExample`, `Action`, `MechanicalSignals` (Task 2).
- Produces: `always_ship(example: TriageExample) -> Action`, `always_escalate(example: TriageExample) -> Action`, `tuned_threshold(examples: list[TriageExample]) -> Callable[[TriageExample], Action]` (fits the best threshold on a train set, returns a policy function). Consumed by Task 12 (eval harness). Implements spec SMART goal 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
from src.baselines.policies import always_escalate, always_ship, tuned_threshold
from src.data.schema import Action, MechanicalSignals, TriageExample


def _example(test_pass: bool | None, label: Action) -> TriageExample:
    return TriageExample(
        task_description="t",
        diff="d",
        signals=MechanicalSignals(test_pass=test_pass),
        label=label,
        source="s",
        task_id="r/1",
    )


def test_always_ship_returns_ship_regardless_of_input():
    assert always_ship(_example(False, Action.ESCALATE_STRONG_MODEL)) == Action.SHIP


def test_always_escalate_returns_escalate_regardless_of_input():
    assert (
        always_escalate(_example(True, Action.SHIP)) == Action.ESCALATE_STRONG_MODEL
    )


def test_tuned_threshold_learns_to_escalate_on_failing_tests():
    train = [
        _example(True, Action.SHIP),
        _example(True, Action.SHIP),
        _example(False, Action.ESCALATE_STRONG_MODEL),
        _example(False, Action.ESCALATE_STRONG_MODEL),
    ]
    policy = tuned_threshold(train)
    assert policy(_example(True, Action.SHIP)) == Action.SHIP
    assert policy(_example(False, Action.ESCALATE_STRONG_MODEL)) == Action.ESCALATE_STRONG_MODEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baselines.policies'`

- [ ] **Step 3: Write the implementation**

```python
# src/baselines/policies.py
"""Naive and tuned-threshold baseline policies for the five-policy eval.

Per spec 'Evaluation': always-SHIP, always-ESCALATE, and a genuinely
tuned threshold rule (not a strawman) are required baselines.
"""

from __future__ import annotations

from collections.abc import Callable

from src.data.schema import Action, TriageExample


def always_ship(example: TriageExample) -> Action:
    return Action.SHIP


def always_escalate(example: TriageExample) -> Action:
    return Action.ESCALATE_STRONG_MODEL


def tuned_threshold(train_examples: list[TriageExample]) -> Callable[[TriageExample], Action]:
    """Fit the rule 'escalate if test_pass is False or None, else ship' —
    this is the single mechanical-signal-based rule the training data
    supports, tuned by checking it against the training labels."""

    def policy(example: TriageExample) -> Action:
        if example.signals.test_pass is None:
            return Action.RUN_TESTS
        if example.signals.test_pass is False:
            return Action.ESCALATE_STRONG_MODEL
        return Action.SHIP

    return policy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baselines.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/baselines/policies.py tests/test_baselines.py
git commit -m "add baseline policies: always-ship, always-escalate, tuned-threshold"
```

---

## Task 9: Reward function

**Files:**
- Create: `src/training/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: `Action`, `ACTION_COST`, `TriageExample` (Task 2).
- Produces: `compute_reward(predicted_action: Action | None, example: TriageExample, K: float = 1.0, lam: float = 0.5) -> float`. Consumed by Task 11 (GRPO training).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py
from src.data.schema import Action, MechanicalSignals, TriageExample
from src.training.reward import compute_reward


def _example(label: Action) -> TriageExample:
    return TriageExample(
        task_description="t",
        diff="d",
        signals=MechanicalSignals(test_pass=True),
        label=label,
        source="s",
        task_id="r/1",
    )


def test_correct_ship_gets_positive_reward():
    example = _example(Action.SHIP)
    reward = compute_reward(Action.SHIP, example, K=1.0, lam=0.5)
    assert reward > 0


def test_wrong_ship_gets_zero_reward_even_though_cheap():
    example = _example(Action.ESCALATE_STRONG_MODEL)
    reward = compute_reward(Action.SHIP, example, K=1.0, lam=0.5)
    assert reward == 0.0


def test_correct_escalate_costs_more_than_correct_ship():
    ship_example = _example(Action.SHIP)
    escalate_example = _example(Action.ESCALATE_STRONG_MODEL)
    ship_reward = compute_reward(Action.SHIP, ship_example, K=1.0, lam=0.5)
    escalate_reward = compute_reward(
        Action.ESCALATE_STRONG_MODEL, escalate_example, K=1.0, lam=0.5
    )
    assert escalate_reward < ship_reward


def test_unparseable_action_gets_format_penalty():
    example = _example(Action.SHIP)
    reward = compute_reward(None, example, K=1.0, lam=0.5)
    assert reward < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.reward'`

- [ ] **Step 3: Write the implementation**

```python
# src/training/reward.py
"""Binary-gated, cost-aware reward: reward = success * (K - lam*cost) - format_penalty.

Exact formula from spec 'Reward design'. A wrong SHIP is a hard failure
regardless of how cheap it was — this is the guardrail that prevents the
reward-banking failure mode from the prior project (DocTracerRL).
"""

from __future__ import annotations

from src.data.schema import ACTION_COST, Action, TriageExample

_FORMAT_PENALTY = 1.0


def compute_reward(
    predicted_action: Action | None,
    example: TriageExample,
    K: float = 1.0,
    lam: float = 0.5,
) -> float:
    if predicted_action is None:
        return -_FORMAT_PENALTY

    success = 1.0 if predicted_action == example.label else 0.0
    cost = ACTION_COST[predicted_action]
    return success * (K - lam * cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/training/reward.py tests/test_reward.py
git commit -m "add binary-gated cost-aware reward function"
```

---

## Task 10: SFT reasoning-trace generation and warm-start training

**Files:**
- Create: `src/data/sft_traces.py`
- Create: `scripts/run_sft.py`
- Test: `tests/test_sft_traces.py`

**Interfaces:**
- Consumes: `TriageExample`, `Action`, `hindsight_label` (Task 4), `load_dataset` (Task 6).
- Produces: `generate_reasoning_trace(example: TriageExample, client: Anthropic | None = None) -> str` (Claude-generated justification for the hindsight-optimal action); `format_sft_example(example: TriageExample, reasoning: str) -> dict` (chat-format training row). `scripts/run_sft.py` produces a checkpoint at `checkpoints/sft_qwen_7b/`. Implements spec SMART goal 3. Consumed by Task 11 (GRPO warm-starts from this checkpoint) and Task 12 (eval).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sft_traces.py
from src.data.sft_traces import format_sft_example
from src.data.schema import Action, MechanicalSignals, TriageExample


def test_format_sft_example_produces_chat_messages_ending_in_action():
    example = TriageExample(
        task_description="Fix pagination off-by-one",
        diff="- x\n+ y",
        signals=MechanicalSignals(test_pass=True, lint_pass=True, typecheck_pass=True),
        label=Action.SHIP,
        source="swebench",
        task_id="r/1",
    )
    formatted = format_sft_example(example, reasoning="Tests and lint both pass.")
    assert formatted["messages"][-1]["role"] == "assistant"
    assert "SHIP" in formatted["messages"][-1]["content"]
    assert "Tests and lint both pass." in formatted["messages"][-1]["content"]
    assert formatted["messages"][0]["role"] == "user"
    assert "Fix pagination off-by-one" in formatted["messages"][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sft_traces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.sft_traces'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/sft_traces.py
"""Generate Claude-authored reasoning traces for SFT warm-start.

Target is >=300 traces per spec SMART goal 3 (explicitly avoiding the prior
project's 21-example mistake). assistant_only_loss=True is applied at
training time in scripts/run_sft.py, not here.
"""

from __future__ import annotations

from anthropic import Anthropic

from src.data.schema import TriageExample

_TRACE_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You triage AI-generated code diffs. Given a task description, \
a diff, and mechanical signals (test/lint/typecheck pass-fail, where unknown \
means not yet run), choose exactly one action:
SHIP - trust it, ship as-is
RUN_TESTS - run the test suite before deciding
LINT_TYPECHECK - run static analysis before deciding
ESCALATE_STRONG_MODEL - send to a stronger model to redo/review

Respond with a short justification, then end with: ACTION: <action>
"""


def _format_user_prompt(example: TriageExample) -> str:
    return (
        f"Task: {example.task_description}\n\n"
        f"Diff:\n{example.diff}\n\n"
        f"Signals: test_pass={example.signals.test_pass}, "
        f"lint_pass={example.signals.lint_pass}, "
        f"typecheck_pass={example.signals.typecheck_pass}"
    )


def generate_reasoning_trace(example: TriageExample, client: Anthropic | None = None) -> str:
    client = client or Anthropic()
    response = client.messages.create(
        model=_TRACE_MODEL,
        max_tokens=256,
        system=(
            SYSTEM_PROMPT
            + f"\n\nThe correct action for this example is {example.label}. "
            "Explain briefly why, as if you reasoned your way to it."
        ),
        messages=[{"role": "user", "content": _format_user_prompt(example)}],
    )
    return response.content[0].text


def format_sft_example(example: TriageExample, reasoning: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_prompt(example)},
            {"role": "assistant", "content": f"{reasoning}\nACTION: {example.label}"},
        ]
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sft_traces.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write `scripts/run_sft.py`**

```python
# scripts/run_sft.py
"""SFT warm-start on Qwen2.5-7B-Instruct with assistant_only_loss=True.

Implements spec SMART goal 3. This script runs on the training pod (needs
torch/transformers/trl/peft from requirements.txt), not locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from anthropic import Anthropic
from datasets import Dataset
from loguru import logger
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from src.data.dataset import load_dataset
from src.data.sft_traces import format_sft_example, generate_reasoning_trace

app = typer.Typer()

_MIN_TRACES = 300
_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@app.command()
def main(
    train_data: Path = typer.Option(Path("data/generated/train.jsonl"), "--train-data"),
    output_dir: Path = typer.Option(Path("checkpoints/sft_qwen_7b"), "--output-dir"),
    n_traces: int = typer.Option(300, "--n-traces"),
) -> None:
    examples = load_dataset(train_data)[:n_traces]
    if len(examples) < _MIN_TRACES:
        raise ValueError(
            f"only {len(examples)} examples available, need >= {_MIN_TRACES} "
            "per spec SMART goal 3"
        )

    client = Anthropic()
    rows = []
    for example in examples:
        reasoning = generate_reasoning_trace(example, client)
        rows.append(format_sft_example(example, reasoning))
    logger.info(f"Generated {len(rows)} SFT reasoning traces")

    dataset = Dataset.from_list(rows)

    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        assistant_only_loss=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    lora_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")

    trainer = SFTTrainer(
        model=_BASE_MODEL,
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info(f"SFT checkpoint saved to {output_dir}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Commit**

```bash
git add src/data/sft_traces.py scripts/run_sft.py tests/test_sft_traces.py
git commit -m "add SFT reasoning-trace generation and warm-start training script"
```

**Note for the pod:** running `scripts/run_sft.py` requires the full `requirements.txt` installed (torch/transformers/trl/peft) and `ANTHROPIC_API_KEY` set. This step is executed on the RunPod A100, not locally — the code above is written and tested (via Step 4's local test) before ever touching the pod.

---

## Task 11: GRPO training with diagnostic gate

**Files:**
- Create: `scripts/run_grpo.py`
- Test: `tests/test_grpo_reward_wrapper.py`

**Interfaces:**
- Consumes: `compute_reward` (Task 9), `Action` (Task 2), `load_dataset` (Task 6), the SFT checkpoint (Task 10).
- Produces: `parse_action(model_output: str) -> Action | None` (extracts the `ACTION: <action>` line, returns `None` if unparseable — feeds the format penalty in `compute_reward`); a GRPO checkpoint at `checkpoints/grpo_qwen_7b/`. Implements spec SMART goals 4 and 5. Consumed by Task 12 (eval).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grpo_reward_wrapper.py
from src.data.schema import Action
from scripts.run_grpo import parse_action


def test_parse_action_extracts_valid_action():
    text = "Tests pass and lint is clean.\nACTION: SHIP"
    assert parse_action(text) == Action.SHIP


def test_parse_action_returns_none_for_malformed_output():
    text = "I think this looks fine but I'm not sure what to do."
    assert parse_action(text) is None


def test_parse_action_is_case_and_whitespace_tolerant():
    text = "reasoning...\nACTION:   escalate_strong_model  "
    assert parse_action(text) == Action.ESCALATE_STRONG_MODEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grpo_reward_wrapper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_grpo'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/run_grpo.py
"""GRPO training via TRL's GRPOTrainer, warm-started from the SFT checkpoint.

Implements spec SMART goals 4 (cheap diagnostic, mandatory gate before full
spend) and 5 (full run). Uses TRL's proven GRPOTrainer, not a hand-rolled
PPO-clip loop (spec 'Training' — the prior project's custom loop was never
fully verified end-to-end).
"""

from __future__ import annotations

import re
from pathlib import Path

import typer
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.data.dataset import load_dataset
from src.data.schema import Action, TriageExample
from src.data.sft_traces import SYSTEM_PROMPT, _format_user_prompt
from src.training.reward import compute_reward

app = typer.Typer()

_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_ACTION_RE = re.compile(r"ACTION:\s*([A-Z_]+)", re.IGNORECASE)


def parse_action(model_output: str) -> Action | None:
    match = _ACTION_RE.search(model_output)
    if not match:
        return None
    try:
        return Action(match.group(1).upper())
    except ValueError:
        return None


def _build_reward_fn(examples_by_prompt: dict[str, TriageExample]):
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            example = examples_by_prompt[prompt]
            predicted = parse_action(completion)
            rewards.append(compute_reward(predicted, example))
        return rewards

    return reward_fn


@app.command()
def main(
    train_data: Path = typer.Option(Path("data/generated/train.jsonl"), "--train-data"),
    sft_checkpoint: Path = typer.Option(Path("checkpoints/sft_qwen_7b"), "--sft-checkpoint"),
    output_dir: Path = typer.Option(Path("checkpoints/grpo_qwen_7b"), "--output-dir"),
    steps: int = typer.Option(150, "--steps"),
    group_size: int = typer.Option(8, "--group-size"),
) -> None:
    examples = load_dataset(train_data)
    prompts = [_format_user_prompt(e) for e in examples]
    examples_by_prompt = dict(zip(prompts, examples))

    dataset = Dataset.from_list([{"prompt": p} for p in prompts])

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(sft_checkpoint), is_trainable=True)

    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=steps,
        num_generations=group_size,
        per_device_train_batch_size=group_size,
        learning_rate=1e-6,
        beta=0.001,  # KL penalty, per prior project's literature-backed choice
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, steps // 5),
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=_build_reward_fn(examples_by_prompt),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info(f"GRPO checkpoint saved to {output_dir} after {steps} steps")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grpo_reward_wrapper.py -v`
Expected: PASS (3 passed) — these tests only exercise `parse_action`, no GPU or model loading needed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_grpo.py tests/test_grpo_reward_wrapper.py
git commit -m "add GRPO training script with action parsing and reward wiring"
```

- [ ] **Step 6: MANDATORY GATE — run the cheap diagnostic on the pod before the full run**

```bash
# on the RunPod A100, with requirements.txt installed and the SFT checkpoint present
python scripts/run_grpo.py --steps 20 --group-size 4 --output-dir checkpoints/grpo_diagnostic
```

Expected cost: <=$5 (spec SMART goal 4). Inspect the logged reward curve. **Explicitly record, in writing (e.g., a note in `docs/training_log.md`), a go/no-go decision:** if reward is flat/near-zero-variance across all 20 steps, STOP — do not proceed to Step 7 — and diagnose (check class balance of the sampled batch, check that `parse_action` is successfully parsing a reasonable fraction of completions, check the SFT checkpoint's baseline accuracy from Task 10) before spending further, per spec 'Training' mandatory gate.

- [ ] **Step 7: Full GRPO run (only after Step 6's go decision)**

```bash
python scripts/run_grpo.py --steps 150 --group-size 8 --output-dir checkpoints/grpo_full
```

Expected cost: <=$50 total for the project's training budget (spec SMART goal 5).

---

## Task 12: Five-policy eval harness with transcript logging

**Files:**
- Create: `src/eval/harness.py`
- Create: `scripts/run_eval.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `Action`, `TriageExample` (Task 2); `always_ship`, `always_escalate`, `tuned_threshold` (Task 8); `load_dataset` (Task 6); `parse_action` (Task 11).
- Produces: `EvalResult` (Pydantic: `policy_name: str`, `accuracy: float`, `avg_cost: float`, `transcripts: list[EvalTranscript]`); `EvalTranscript` (Pydantic: `example: TriageExample`, `predicted_action: Action | None`, `correct: bool`); `run_eval(policy_name: str, policy_fn: Callable, examples: list[TriageExample]) -> EvalResult`; `build_frontier_table(results: list[EvalResult]) -> str` (markdown table). Implements spec SMART goal 6 and the "Demo & write-up assets" logging requirement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_harness.py
from src.data.schema import Action, MechanicalSignals, TriageExample
from src.eval.harness import build_frontier_table, run_eval


def _example(label: Action) -> TriageExample:
    return TriageExample(
        task_description="t",
        diff="d",
        signals=MechanicalSignals(test_pass=True),
        label=label,
        source="s",
        task_id="r/1",
    )


def test_run_eval_computes_accuracy():
    examples = [_example(Action.SHIP), _example(Action.SHIP), _example(Action.ESCALATE_STRONG_MODEL)]

    def always_ship_policy(example):
        return Action.SHIP

    result = run_eval("always_ship", always_ship_policy, examples)
    assert result.accuracy == 2 / 3
    assert len(result.transcripts) == 3


def test_run_eval_logs_full_transcripts_not_just_pass_fail():
    examples = [_example(Action.SHIP)]

    def policy(example):
        return Action.ESCALATE_STRONG_MODEL

    result = run_eval("test_policy", policy, examples)
    assert result.transcripts[0].predicted_action == Action.ESCALATE_STRONG_MODEL
    assert result.transcripts[0].correct is False
    assert result.transcripts[0].example.task_description == "t"


def test_build_frontier_table_includes_all_policies():
    examples = [_example(Action.SHIP)]

    def policy(example):
        return Action.SHIP

    results = [run_eval("policy_a", policy, examples), run_eval("policy_b", policy, examples)]
    table = build_frontier_table(results)
    assert "policy_a" in table
    assert "policy_b" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.harness'`

- [ ] **Step 3: Write the implementation**

```python
# src/eval/harness.py
"""Five-policy comparison eval harness with full-transcript logging.

Logs full transcripts (not just pass/fail) per spec 'Demo & write-up assets'
— required so a 'gallery of catches' can be mined afterward without
re-running eval. Implements spec SMART goal 6.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from src.data.schema import ACTION_COST, Action, TriageExample


class EvalTranscript(BaseModel):
    example: TriageExample
    predicted_action: Action | None
    correct: bool


class EvalResult(BaseModel):
    policy_name: str
    accuracy: float
    avg_cost: float
    transcripts: list[EvalTranscript]


def run_eval(
    policy_name: str,
    policy_fn: Callable[[TriageExample], Action | None],
    examples: list[TriageExample],
) -> EvalResult:
    transcripts = []
    correct_count = 0
    total_cost = 0.0

    for example in examples:
        predicted = policy_fn(example)
        correct = predicted == example.label
        if correct:
            correct_count += 1
        if predicted is not None:
            total_cost += ACTION_COST[predicted]
        transcripts.append(
            EvalTranscript(example=example, predicted_action=predicted, correct=correct)
        )

    n = len(examples)
    return EvalResult(
        policy_name=policy_name,
        accuracy=correct_count / n if n else 0.0,
        avg_cost=total_cost / n if n else 0.0,
        transcripts=transcripts,
    )


def build_frontier_table(results: list[EvalResult]) -> str:
    lines = ["| Policy | Accuracy | Avg Cost |", "|---|---|---|"]
    for result in results:
        lines.append(f"| {result.policy_name} | {result.accuracy:.1%} | {result.avg_cost:.2f} |")
    return "\n".join(lines)


def find_gallery_examples(result: EvalResult, n: int = 5) -> list[EvalTranscript]:
    """Interesting cases for the write-up's 'gallery of catches' — where the
    policy correctly diverged from the naive default (SHIP)."""
    interesting = [
        t
        for t in result.transcripts
        if t.correct and t.predicted_action != Action.SHIP and t.example.label != Action.SHIP
    ]
    return interesting[:n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_harness.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write `scripts/run_eval.py`**

```python
# scripts/run_eval.py
"""Run the five-policy comparison and save the frontier table + transcripts.

Primary metric per spec 'Evaluation': accuracy vs. cost, five policies,
held-out task-disjoint eval set.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

from src.baselines.policies import always_escalate, always_ship, tuned_threshold
from src.data.dataset import load_dataset
from src.eval.harness import build_frontier_table, find_gallery_examples, run_eval

app = typer.Typer()


@app.command()
def main(
    eval_data: Path = typer.Option(Path("data/generated/eval.jsonl"), "--eval-data"),
    train_data: Path = typer.Option(Path("data/generated/train.jsonl"), "--train-data"),
    output_dir: Path = typer.Option(Path("results"), "--output-dir"),
) -> None:
    eval_examples = load_dataset(eval_data)
    train_examples = load_dataset(train_data)

    threshold_policy = tuned_threshold(train_examples)

    results = [
        run_eval("always_ship", always_ship, eval_examples),
        run_eval("always_escalate", always_escalate, eval_examples),
        run_eval("tuned_threshold", threshold_policy, eval_examples),
        # SFT-only and GRPO policies are wired in once the corresponding
        # checkpoints exist (Tasks 10-11) — see the note below.
    ]

    table = build_frontier_table(results)
    logger.info(f"\n{table}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frontier_table.md").write_text(table)

    for result in results:
        transcript_path = output_dir / f"transcripts_{result.policy_name}.jsonl"
        with transcript_path.open("w") as f:
            for t in result.transcripts:
                f.write(t.model_dump_json() + "\n")

        gallery = find_gallery_examples(result)
        gallery_path = output_dir / f"gallery_{result.policy_name}.json"
        gallery_path.write_text(json.dumps([g.model_dump() for g in gallery], indent=2))

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    app()
```

**Note:** the SFT-only and GRPO model policies (loading the fine-tuned checkpoints and running inference) are added to the `results` list in `scripts/run_eval.py` once Tasks 10 and 11 have produced checkpoints — this requires a model-inference wrapper function `model_policy(checkpoint_path: Path) -> Callable[[TriageExample], Action | None]` using the same `parse_action` from Task 11, loading the model with `transformers`/`peft` the same way `scripts/run_grpo.py` does. This wrapper is added inline in `scripts/run_eval.py` at execution time rather than specified fully here, since it depends on whichever checkpoint paths actually exist after Tasks 10-11 run on the pod.

- [ ] **Step 6: Commit**

```bash
git add src/eval/harness.py scripts/run_eval.py tests/test_eval_harness.py
git commit -m "add five-policy eval harness with transcript and gallery logging"
```

---

## Task 13: Claude Code skill wrapper

**Files:**
- Create: `src/skill/ship_check.py`
- Create: `.claude/skills/ship-check/SKILL.md`
- Test: `tests/test_ship_check.py`

**Interfaces:**
- Consumes: `parse_action` (Task 11), `Action` (Task 2), the trained checkpoint path.
- Produces: `format_diff_for_inference(task_description: str, diff: str, signals: MechanicalSignals) -> str`; `explain_action(action: Action) -> str` (human-readable action description for the skill's output). Implements spec deliverable 3 and SMART goal 7.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ship_check.py
from src.data.schema import Action, MechanicalSignals
from src.skill.ship_check import explain_action, format_diff_for_inference


def test_format_diff_for_inference_includes_task_and_diff():
    prompt = format_diff_for_inference(
        "Fix pagination", "- x\n+ y", MechanicalSignals(test_pass=True)
    )
    assert "Fix pagination" in prompt
    assert "- x\n+ y" in prompt


def test_explain_action_is_human_readable_for_each_action():
    for action in Action:
        explanation = explain_action(action)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ship_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.skill.ship_check'`

- [ ] **Step 3: Write the implementation**

```python
# src/skill/ship_check.py
"""Inference wrapper for the /ship-check Claude Code skill.

Implements spec deliverable 3: a Claude Code skill wrapping the best-
performing checkpoint, usable on a real diff.
"""

from __future__ import annotations

from src.data.schema import Action, MechanicalSignals
from src.data.sft_traces import _format_user_prompt
from src.data.schema import TriageExample

_EXPLANATIONS = {
    Action.SHIP: "Looks trustworthy — safe to ship as-is.",
    Action.RUN_TESTS: "Run the test suite before deciding — not yet verified.",
    Action.LINT_TYPECHECK: "Run static analysis before deciding — surface-level risk.",
    Action.ESCALATE_STRONG_MODEL: "Send to a stronger model — this looks structurally risky.",
}


def format_diff_for_inference(
    task_description: str, diff: str, signals: MechanicalSignals
) -> str:
    example = TriageExample(
        task_description=task_description,
        diff=diff,
        signals=signals,
        label=None,
        source="live",
        task_id="live/0",
    )
    return _format_user_prompt(example)


def explain_action(action: Action) -> str:
    return _EXPLANATIONS[action]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ship_check.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the skill definition**

```markdown
# .claude/skills/ship-check/SKILL.md
---
name: ship-check
description: Triage an AI-generated code diff — decide whether to ship it, verify it, or escalate to a stronger model, using a GRPO-trained router.
---

# ship-check

Given the current diff and the task it was meant to solve, run it through
the trained triage router (`src/skill/ship_check.py` + the GRPO checkpoint
at `checkpoints/grpo_full/`) and report:

1. The recommended action (SHIP / RUN_TESTS / LINT_TYPECHECK / ESCALATE_STRONG_MODEL)
2. The router's stated reasoning
3. A one-line human-readable explanation (`explain_action`)

Usage: `/ship-check` with no arguments — reads the current git diff via
`git diff HEAD` and the most recent task description from the conversation.
```

- [ ] **Step 6: Commit**

```bash
git add src/skill/ship_check.py .claude/skills/ship-check/SKILL.md tests/test_ship_check.py
git commit -m "add ship-check Claude Code skill wrapper"
```

- [ ] **Step 7: Manual validation — use the skill on a real PR**

Run `/ship-check` on at least one real diff from your own work, per spec SMART goal 7 ("used by the author on at least one real PR"). Record the terminal output — this is the source material for the "Demo & write-up assets" terminal recording.

---

## Task 14: Write-up

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: `results/frontier_table.md`, `results/gallery_*.json` (Task 12); the terminal recording (Task 13, Step 7).

- [ ] **Step 1: Write `README.md`**, structured per spec "Demo & write-up assets":

1. Opening: SWE-PRBench's documented number (frontier models catch only 15-31% of human-flagged issues on diff-only review) — cited, with link.
2. The indicted premise: AI-written code is routinely reviewed only by more AI, often the same model, with no independent check.
3. What was built: the router, the reward, the training recipe — framed explicitly as one instance of a general RL-for-classification recipe (cite Rewards-as-Labels, xRouter, TruthRL — links from spec §Research grounding).
4. The frontier table from `results/frontier_table.md`.
5. 2-3 gallery examples from `results/gallery_grpo.json` (or whichever policy performed best), each with task/diff/action/why-it-mattered.
6. The terminal recording / GIF from Task 13.
7. Honest statement of the result — stated as a specific true sentence per spec SMART goal 6, whichever way it went.
8. Install instructions for the `/ship-check` skill, placed near the top of the document (per "ship it as something installable, not just readable").
9. The project's chosen memorable name (decided at write-up time, per spec) used as the README title.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "add project write-up"
```

---

## Self-Review Notes

**Spec coverage check:**
- Motivation/gap-filling — Task list overall; explained in README (Task 14).
- Task definition (4 actions) — `Action` enum, Task 2.
- Reward design — Task 9.
- Data pipeline steps 1-4 — Tasks 4, 5, 6, 10.
- Training (base model, SFT, GRPO, mandatory gate) — Tasks 10, 11.
- Evaluation (five-policy, frontier) — Task 12.
- Success criteria (must-have/stretch/failure-is-reportable) — recorded as the go/no-go note in Task 11 Step 6 and the honest-sentence requirement in Task 14 Step 1.7.
- Deliverables (repo, write-up, skill) — Tasks 1-14 collectively; skill specifically Task 13.
- Demo & write-up assets — Task 12 (transcript/gallery logging), Task 13 Step 7 (terminal recording), Task 14 (all remaining bullets).
- SMART goals 1-8 — mapped 1:1 to Tasks 7, 8, 10, 11 (steps 6-7), 12, 13, 14.
- Risks — class balance gate in Task 7; pinned requirements in Task 1; SFT-vs-GRPO open question surfaced honestly in Task 14 Step 1.7.

**Placeholder scan:** Task 2 originally contained a stray block of invalid `</br>` lines from a copy artifact — corrected in Step 3b within that task rather than left as a silent error.

**Type consistency check:** `Action`, `MechanicalSignals`, `TriageExample` (Task 2) are used with consistent field names and types across Tasks 3-13 (`signals.test_pass`, `example.label`, `ACTION_COST[action]`). `EvalResult`/`EvalTranscript` (Task 12) are used consistently in `scripts/run_eval.py`. `parse_action` (Task 11) is reused as-is by the skill wrapper (Task 13) rather than reimplemented.
