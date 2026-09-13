"""Create a separate demo vault using scripted models; no GPU, API, or trained model."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research.agent import run_question  # noqa: E402
from src.research.explainer import run_explanation  # noqa: E402
from src.research.tools_runtime import ResearchTools  # noqa: E402
from src.research.vault import export_explanation, import_vault  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for demo files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    vault, snapshot = args.output / "Vault", args.output / "snapshot"
    shutil.copytree(ROOT / "examples/research-vault", vault)
    import_vault(vault, "Research", snapshot)
    hit = ResearchTools(snapshot / "corpus").search_papers("original paper passage", 1)[0]
    claim = "The personal note recommends checking the original passage before trusting a claim."

    class ScriptedRetriever:
        config = {"backend": "scripted_demo", "model": "none", "seed": 0}
        step = 0

        def act(self, observation):
            self.step += 1
            if self.step == 1:
                return json.dumps({"action": "search_papers", "arguments": {
                    "query": "original paper passage", "top_k": 1}})
            if self.step == 2:
                return json.dumps({"action": "passage", "arguments": {
                    "doc_id": hit["doc_id"], "start": hit["start"],
                    "length": hit["end"] - hit["start"]}})
            return json.dumps({"action": "submit", "answer": {
                "claims": [{"text": claim, "evidence": [
                    {key: hit[key] for key in ("doc_id", "start", "end")}
                ]}], "recommendation": "Inference: open the original passage to check a claim.",
                "limitations": ["Personal commentary; no published result established."]}})

    class ScriptedExplainer:
        config = {"backend": "scripted_demo", "model": "none"}

        def explain(self, packet):
            return json.dumps({
                "answer": "This is a scripted integration demo; no Qwen or larger model ran.\n\n"
                "Your example note recommends checking the original paper passage before trusting "
                "a research claim [E1]. Inference: when you read an explanation, open its evidence "
                "link and compare the quoted passage with the claim being made.",
                "claims": [{"text": claim, "evidence_ids": ["E1"]}],
                "limitations": ["This demonstrates file import, search, and cited note export.",
                                "It does not measure model research quality."]})

    retriever = args.output / "retriever.json"
    run_question(snapshot, {"id": "vault_demo", "split": "development",
                           "question": "How do my notes suggest checking research claims?"},
                 ScriptedRetriever(), retriever, max_steps=3)
    explanation = args.output / "explanation.json"
    run_explanation(retriever, ScriptedExplainer(), explanation)
    exported = export_explanation(explanation, snapshot, vault, "Answers/Checking claims demo.md")
    print(f"Scripted demo complete. Open {vault} as an Obsidian vault.")
    print(f"Exported note: {exported}")


if __name__ == "__main__":
    main()
