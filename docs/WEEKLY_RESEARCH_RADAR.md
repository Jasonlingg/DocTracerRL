# Weekly AI research radar

## The product

Every week, the system finds new papers in AI areas selected by the user, removes duplicates,
ranks the candidates, investigates the most relevant work, and writes an evidence-backed digest
into an Obsidian vault. Claude, GPT, or another main assistant can query the same library through
an MCP server.

The useful output is not a feed of paper titles. It answers:

- What changed this week?
- Which papers matter for my current projects, and why?
- How does each result relate to work already in my vault?
- What evidence in the paper supports the summary?
- What remains uncertain, missing, or unverified?

## System boundary

```text
Scheduled discovery
  arXiv / reviewed source APIs
             |
             v
Candidate store -> deduplication -> topic ranking
                                      |
                                      v
                           Qwen code-execution worker
                         search / read / extract / compare
                                      |
                                      v
                       evidence packets + exact source spans
                                      |
                       +--------------+--------------+
                       |                             |
                       v                             v
              Obsidian weekly digest          MCP research server
                                                /            \
                                           Claude             GPT
```

Discovery, investigation, presentation, and storage are separate stages. The scheduler fetches
new candidates. Qwen investigates a frozen candidate corpus by writing Python. The MCP caller gets
compact evidence packets rather than Qwen's raw execution transcript. Obsidian stores the durable
paper notes, topic pages, and weekly digests.

## Why Qwen is inside the MCP service

Claude or GPT should not micromanage every `search()` and `read()` call. The MCP server exposes
high-level operations, while Qwen performs the repeated local work:

- `get_weekly_digest(topic, week)`
- `search_research_library(query, filters)`
- `investigate_research_question(question, topic, budget)`
- `get_paper_evidence(paper_id, claim)`

For example, the main assistant asks `investigate_research_question`. The server starts an isolated
Qwen episode, Qwen writes Python against the frozen library, and the server returns claims, source
IDs, exact passages, limitations, and the saved trajectory. This keeps the small model's trained
skill useful while allowing any capable host model to explain or apply the result.

Read-only research tools are the first MCP surface. Writing a note to Obsidian is a separate,
explicit operation so a host cannot silently modify the vault while answering a question.

## Weekly pipeline

1. Read topic profiles containing positive examples, exclusions, keywords, venues, authors, and
   project context.
2. Fetch papers published or revised since the previous successful run. Store provider IDs,
   versions, timestamps, URLs, and retrieval time.
3. Deduplicate by arXiv ID, DOI, and normalized title. Preserve revisions rather than treating a
   revised paper as unrelated.
4. Rank candidates using metadata and retrieval first. Apply an expensive model judgment only to
   the bounded shortlist.
5. Freeze the selected abstracts/full text into an immutable weekly corpus.
6. Let Qwen investigate each candidate and compare it with relevant existing vault notes.
7. Validate cited spans mechanically and retain unsupported/uncertain labels.
8. Generate a weekly digest and individual paper notes. Never overwrite a prior week's record.
9. Update the MCP search index only after the snapshot and notes are complete.

If a weekly run fails, it records the failure and leaves the last successful index intact. Re-runs
use the same candidate IDs and corpus revision so results can be reproduced.

## Obsidian layout

```text
AI Research/
  Topics/
    Retrieval Agents.md
    Small Model Training.md
  Papers/
    2026/
      <paper-id> - <short title>.md
  Weekly/
    2026-W38.md
  Questions/
    Open Questions.md
```

Paper notes should contain stable IDs, version/date, source URL, topic tags, a short contribution
summary, relationships to saved papers, exact supporting passages, limitations, and review status.
Ordinary Markdown links and frontmatter are sufficient initially; a separate graph database is not
required to obtain Obsidian backlinks and topic navigation.

## What exists

- Versioned paper discovery and snapshot code.
- Markdown/PDF vault import with immutable corpus hashes.
- The `search()`, `read()`, `extract()`, and related Python tools.
- A multi-turn Qwen policy and persistent local worker per episode.
- MuSiQue base/SFT/GRPO evaluation and saved code trajectories.
- Exact-span machinery in the earlier research path that can be reused at the output boundary.

## What remains

1. Topic-profile schema and saved discovery cursor.
2. Candidate database with deduplication and revision handling.
3. A code-execution question format for paper investigation and comparison.
4. Exact-span citations in the executable-code submission protocol.
5. Weekly digest and paper-note writers.
6. Read-only MCP server wrapping the Qwen worker and saved artifacts.
7. A scheduler that can run while the laptop is closed.
8. Evaluation over several historical weeks before enabling unattended updates.

## Evaluation before automation

Replay several past weeks for two narrowly defined AI topics. Human-label a bounded candidate pool
for relevance and review the resulting evidence packets.

**Hypothesis:** the trained Qwen worker can reduce the candidate set to a useful weekly shortlist
and produce more supported evidence packets than base Qwen under the same retrieval and token
budget.

**Expected signal:** improved precision at 10, supported-claim rate, and successful completion,
with fewer repeated searches and execution failures. Also record discovery recall, citation
accuracy, latency, GPU time, and the number of papers requiring manual correction.

**Decision rule:** schedule the pipeline only if it finds nearly all human-selected important
papers in the frozen pool and most digest claims are supported by the cited spans. If discovery
recall is weak, fix sources and topic profiles. If ranking is weak, improve retrieval/ranking. If
citations are real but do not support the claims, improve the investigation data and verifier.

## Delivery sequence

1. Implement topic profiles, weekly discovery state, and candidate deduplication on CPU.
2. Build one frozen historical-week fixture and a small reviewed relevance set.
3. Add exact-span submission to the code-execution environment.
4. Compare base and SFT Qwen on that fixture.
5. Generate Obsidian notes from accepted evidence packets.
6. Expose saved search, evidence, and digest operations through a local MCP server.
7. Add authenticated remote MCP only when a remote host is needed.
8. Move the weekly job to an always-on scheduler after replay tests pass.
