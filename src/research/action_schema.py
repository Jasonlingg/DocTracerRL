"""Generation contract for research-tools-v2; source support remains a separate check."""

import json

from jsonschema import Draft202012Validator


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


_STRING = {"type": "string"}
_DOC_ID = {"type": "string", "minLength": 1}
_EVIDENCE = _object({
    "doc_id": _DOC_ID,
    "start": {"type": "integer", "minimum": 0},
    "end": {"type": "integer", "minimum": 1},
})
_ANSWER = _object({
    "claims": {"type": "array", "items": _object({
        "text": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "minItems": 1, "items": _EVIDENCE},
    })},
    "recommendation": _STRING,
    "limitations": {"type": "array", "items": _STRING},
})
_TOOLS = {
    "papers": _object({}),
    "search_papers": _object({
        "query": {"type": "string", "minLength": 1},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
    }, ["query"]),
    "paper": _object({"doc_id": _DOC_ID}),
    "passage": _object({
        "doc_id": _DOC_ID,
        "start": {"type": "integer", "minimum": 0},
        "length": {"type": "integer", "minimum": 1, "maximum": 3000},
    }, ["doc_id"]),
}
ACTION_SCHEMA = {"anyOf": [
    _object({"action": {"const": name}, "arguments": arguments})
    for name, arguments in _TOOLS.items()
] + [_object({"action": {"const": "submit"}, "answer": _ANSWER})]}
_VALIDATOR = Draft202012Validator(ACTION_SCHEMA)


def validate_structured_action(raw: str) -> dict:
    """Fail closed if a server ignores the requested schema; never repair model output."""
    action = json.loads(raw)
    _VALIDATOR.validate(action)
    return action
