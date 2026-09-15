# Use an Obsidian vault as the research library

This integration works through ordinary folders and Markdown. No Obsidian plugin is required.
Import and local harness checks run on your CPU. The active agent uses the executable Python tools
in `src/env/`; running Qwen still requires a model endpoint or GPU. The older JSON-action research
and explainer commands below remain available but are not the current training target.

## Use the code-execution environment

Import a frozen snapshot, then pass its `corpus/` directory to the standard evaluation runner:

```bash
python3 scripts/research_vault.py import \
  --vault '/path/to/My Vault' --collection Research \
  --output out/research/my-vault-snapshot-1

python3 scripts/run_eval.py \
  --corpus out/research/my-vault-snapshot-1/corpus \
  --questions data/my-vault/questions.json \
  --policy qwen_sft_policy --output out/my-vault-sft.json
```

Each question uses the existing multi-step loop: Qwen emits Python, the sandbox executes it, and
the observation returns to Qwen. A typical trajectory calls `search(...)`, saves a document ID,
uses `read(...)` or `extract(...)`, and finally emits `SUBMIT:`. A private snapshot belongs under
`out/`, which is ignored by Git. Do not commit the imported note text.

## Try the local example

To create a separate demo vault with an exported answer and its exact source links:

```bash
python3 scripts/research_vault_demo.py --output out/research/obsidian-demo
```

Open `out/research/obsidian-demo/Vault` in Obsidian. The demo uses scripted responses explicitly
labeled as a demo; it makes no API calls and establishes no model capability.

Open `examples/research-vault` as a folder vault in Obsidian, or read its files in any editor.
The two example notes contain personal preferences and open questions, not paper evidence.

```bash
python3 scripts/research_vault.py import \
  --vault examples/research-vault --collection Research \
  --output out/research/vault-demo
python3 scripts/research.py inspect \
  --snapshot out/research/vault-demo --query 'original paper passage' --top-k 2
```

The snapshot directory must be new and outside the vault. It preserves source bytes, extraction
metadata, text offsets, and checksums. Import another snapshot after editing or adding documents;
there is no background watcher yet. Never update the frozen corpus used by an earlier evaluation.

## Connect your own collection

Choose an existing vault and a relative folder, for example `/path/to/My Vault` and `Research`.
Only `.md` and `.pdf` files inside that selected folder are imported recursively. The importer skips
hidden files, symlinks, and notes marked `generated_by: rlm-explorer`. It does not follow wikilinks
outside the collection or change your existing notes.

For PDF import, install the optional dependency in your project environment:

```bash
python3 -m pip install '.[vault]'
python3 scripts/research_vault.py import \
  --vault '/path/to/My Vault' --collection Research \
  --output out/research/my-vault-snapshot-1
```

PDFs use page-by-page text extraction with pypdf; the installed parser version is recorded.
Empty pages are listed in `manifest.json`. Image-only PDFs fail explicitly and need OCR elsewhere.
Tables, diagrams, and equations may be missing or distorted. PDF file type does not verify that a
document is a published paper. Markdown is labeled `personal_note`; PDFs are `pdf_document`.
Generated explanations keep their marker when moved and are excluded from later imports. Imported
third-party AI notes without the marker remain personal commentary; add the marker to exclude them.

## Ask, explain, save

The new snapshot plugs into the existing multi-step research command:

```bash
python3 scripts/research.py ask \
  --snapshot out/research/my-vault-snapshot-1 \
  --question 'What evidence in these documents addresses my retrieval problem?' \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-8B \
  --revision MODEL_COMMIT --server-hardware 'ACTUAL SERVING CONFIGURATION' \
  --output out/research/my-question.json

python3 scripts/research_explain.py \
  --retriever-run out/research/my-question.json \
  --endpoint http://localhost:9000/v1 --model EXPLAINER_MODEL \
  --revision EXPLAINER_REVISION --server-hardware 'ACTUAL SERVING CONFIGURATION' \
  --output out/research/my-explanation.json

python3 scripts/research_vault.py export \
  --vault '/path/to/My Vault' --snapshot out/research/my-vault-snapshot-1 \
  --explanation out/research/my-explanation.json --note 'Answers/Retrieval question.md'
```

Replace the endpoint/model placeholders with your actual serving configuration. A remote model
receives selected passages; use local endpoints if your material must stay on the computer.

The exported note contains the explanation, individual claims, linked evidence, exact quotations,
source types, PDF page links where available, limitations, and an unreviewed AI-draft label.
Export never overwrites an existing note. It rechecks quotations against the frozen snapshot instead
of trusting the artifact's stored `checks` flag. If the source file has changed, the note warns that
the link now points at a different version and retains the original frozen quotation.

## Scope and validation

This is a manual import/search/export loop for a small collection. Search still scans the frozen
text with the existing lexical scorer. Hybrid indexing, incremental refresh, a chat interface, and
an independently validated trained Qwen adapter remain future work.

Offline tests cover a complete scripted search/submit/explain/export loop, actual PDF text
extraction and blank-page diagnostics, source-type propagation, generated-note exclusion, path
isolation, immutable revisions, changed-source warnings, and refusal to export forged quotations.
Scripted model tests prove the integration works, not that Qwen's research quality improved.
