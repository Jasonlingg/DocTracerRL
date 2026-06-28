"""Parse raw Claude/Haiku output into executable Python or a SUBMIT line.

Haiku-specific quirks this handles that the simpler Qwen `clean_action`
(qwen_common.py) doesn't need to: XML function_calls tags, prose mixed in
with code, and multi-step dumps in a single response.
"""

from __future__ import annotations

import re


def _truncate_to_first_step(code: str) -> str:
    """If the model generated multiple steps, keep only the first one.

    Haiku often outputs '# Step 1 ... # Step 2 ... # Step 3 ...' as one block.
    Code for later steps references variables that don't exist yet (because
    the agent hasn't seen earlier steps' output). This is not a policy choice
    we're constraining — it's broken code that will SyntaxError/NameError.
    """
    lines = code.split("\n")
    step_markers = []
    for i, line in enumerate(lines):
        if re.match(r"^#\s*Step\s+\d", line.strip(), re.IGNORECASE):
            step_markers.append(i)

    # Only truncate if there are 2+ step markers (multi-step dump) — cut at the second marker
    if len(step_markers) >= 2:
        code = "\n".join(lines[: step_markers[1]]).strip()

    return code


def clean_action(text: str) -> str:
    """Extract executable Python from model output, stripping fences, XML, and prose."""
    stripped = text.strip()

    # If there's a SUBMIT line anywhere, extract and return it
    # (model is ready to answer — don't try to run code too)
    submit_match = re.search(
        r"(SUBMIT:\s*.*?CITATIONS:\s*\[.*?\])",
        stripped,
        re.DOTALL | re.IGNORECASE,
    )
    if not submit_match:
        submit_match = re.search(
            r"(SUBMIT:\s*.+)",
            stripped,
            re.IGNORECASE,
        )
    if submit_match:
        return submit_match.group(1).strip()

    # Strip markdown code fences
    if stripped.startswith("```python") and stripped.endswith("```"):
        stripped = stripped[len("```python"):][:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1]).strip()

    # Strip XML function_calls (Haiku sometimes generates these)
    if "<function_calls>" in stripped or "<invoke" in stripped:
        stripped = re.sub(r"</?function_calls>", "", stripped)
        stripped = re.sub(r"</?invoke[^>]*>", "", stripped)
        stripped = re.sub(r"</?parameter[^>]*>", "", stripped)

    # Extract code from markdown fences embedded in prose
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", stripped, re.DOTALL)
    if code_blocks:
        stripped = "\n".join(code_blocks).strip()

    # Filter: keep only lines that look like Python code, drop prose
    lines = stripped.split("\n")
    code_lines = []
    for line in lines:
        s = line.strip()
        # Keep blank lines (they're valid Python)
        if not s:
            code_lines.append(line)
            continue
        # Keep lines that look like code
        if re.match(
            r"^("
            r"#|"                           # comments
            r"[a-zA-Z_]\w*\s*[=(.\[]|"      # assignment, call, attribute, index
            r"from |import |"               # imports
            r"print\(|"                     # print
            r"for |if |elif |else:|while |" # control flow
            r"def |class |"                 # definitions
            r"return |yield |"              # returns
            r"try:|except |finally:|"       # exception handling
            r"with |"                       # context managers
            r"raise |assert |"              # raise/assert
            r"pass|break|continue|"         # simple statements
            r"\)|"                          # closing paren (continuation)
            r"\]|"                          # closing bracket
            r"\}|"                          # closing brace
            r"\"\"\"|'''|"                  # docstrings
            r"@"                            # decorators
            r")",
            s,
        ):
            code_lines.append(line)
        # else: drop the line (it's prose)

    result = "\n".join(code_lines).strip()
    result = result if result else stripped

    # Haiku often generates ALL steps at once (# Step 1 ... # Step 2 ...).
    # Truncate to just the first step to avoid running code that depends
    # on output the agent hasn't seen yet.
    result = _truncate_to_first_step(result)

    return result
