# Compiling Natural-Language Specs into Executable Programs with RL — Design Spec

*Working title. A short, memorable project name should be chosen before the write-up ships — "the extraction compiler" is a description, not a name.*

## One-line pitch

Train a small open model (Qwen2.5-Coder-7B, LoRA) with GRPO to take a natural-language request and a real web page, explore that page by writing and executing code in a sandbox, and emit a **reusable Python extraction program** that is scored on pages from websites it has never seen. The artifact is deterministic code with zero inference cost, not a model call per page.

**Do not describe this project as "web scraping."** The task is compiling a fuzzy natural-language spec into an executable program; web extraction is the instantiation chosen because verification is exact and free there. The framing determines whether the right people read past the first line.

## Motivation

This closes a specific, verifiable gap in the author's track record. The resume reads:

> "Designed **RL environments** … where **Anthropic Claude models** autonomously solve ML engineering tasks via **code execution in sandboxed containers**, evaluated against **verifiable reward signals**." — Preference Model

That is nearly this project, except the actor was a model someone else trained. Every ML bullet on the resume is environment design, pipeline, eval, or infrastructure. There is no line stating that the author trained a policy and it measurably improved. Assort Health contributes quality numbers (40–50% → 80% entity resolution, ~95% garbled-entity recognition) but those came from data-pipeline work, not from training.

The narrative this produces: *"I designed these environments at work; then I built one myself and trained the policy in it."* That strengthens the existing Preference Model line retroactively rather than merely sitting beside it.

A prior project (DocTracerRL / rlm-explorer) attempted GRPO on multi-hop QA and stalled with a flat reward curve. Three causes were diagnosed: sparse trajectory-level reward (one fuzzy F1 across a ten-step episode), a fatally undersized SFT warm-start (21 examples), and a task that likely exceeded a 7B LoRA model's capability ceiling. This project is designed against all three.

## Why this task, not another one

Rejected during brainstorming, with reasons recorded so they are not re-litigated:

- **Code-triage router** (fully spec'd, `2026-08-27-code-triage-router-design.md`) — a single-shot classification decision wearing RL's mechanics. Killed by a standing constraint: *"I don't think RL is the right tool for classifier stuff."* Independently confirmed as a documented failure pattern — RLVR models "abandon rule induction and enumerate instance-level labels" when a cheap correlated signal exists.
- **Synthetic messy-data tasks** (tabular aggregation, record linkage, format repair) — viable, but the difficulty would be self-generated, and a skeptic correctly notes that generating both the problem and the answer means generating the difficulty. Also: PAW's own FuzzyBench already covers this territory (800+ categories including format conversion, parsing, fuzzy matching, log triage), and it is unreleased, so it is ground we cannot measure ourselves against.
- **Building on / retraining PAW** — blocked on artifact availability, not merit. Neither weights, training code, nor the 10M-example FuzzyBench dataset are public. Retraining a 4B hypernetwork compiler is also orders of magnitude beyond budget, and would be supervised rather than RL.
- **Countdown/math GRPO reproduction, chess, poker, negotiation, forecasting** — rejected in prior sessions for lack of differentiation, capability-ceiling risk, or absent personal interest.

Web extraction was selected because:

1. **Exploration is structurally forced.** A real page's DOM cannot be known from the prompt, and pages are too large to read wholesale. The agent must poke at the page with code and adapt to what comes back. This is genuine multi-step RL, not codegen with extra turns.
2. **Ground truth is exact, free, and self-renewing.** See Data pipeline.
3. **The generalization axis is concrete and demonstrable.** "The site got redesigned and the parser still worked" needs no explanation to any audience.
4. **It is within a code model's competence.** Writing selectors and regexes over a discovered structure is far closer to a code model's training distribution than judging whether a diff is subtly wrong.
5. **The cost argument is real and quantifiable.** The industry answer today is an LLM call per page, forever.

## Why this avoids the prior project's failure mode

| DocTracerRL failure cause | How this project differs |
|---|---|
| Sparse, fuzzy trajectory-level reward (one F1 across 10 steps) | Reward is dense: field-level F1 across many records × many held-out pages. A parser that gets names right and prices wrong scores a meaningful, non-zero fraction, so small improvements are visible to the gradient |
| SFT warm-start was 21 hand-collected examples | Warm-start is RAFT (rejection sampling): sample many programs per task, keep the ones the verifier passes. Auto-labeled, unlimited scale, zero human labeling |
| Task may have exceeded the base model's capability ceiling | Task is code generation over an inspectable structure, with difficulty directly tunable via a curriculum on how far held-out sites drift from training sites |
| No cheap early signal that training was failing | A mandatory ≤$5, 15–20 step diagnostic gate before committing the full budget, carried over from the triage spec |

## Research grounding

- **[SCRIBES (Meta, arXiv 2510.01832)](https://arxiv.org/abs/2510.01832)** — the closest prior art and the honest anchor. GRPO-trains an LLM to write reusable extraction scripts from HTML, with the same "compile once, avoid per-page LLM calls" argument and computed FLOPS savings. **This project's differentiation is specific and checkable:** (a) SCRIBES uses Qwen2.5-14B/32B and Llama-3.3-70B with full-parameter FSDP finetuning; this is a 7B LoRA on a single pod. (b) SCRIBES is one-shot — model sees a page, emits a script, no iterative exploration; this project's agent explores by executing code and observing real output. (c) **SCRIBES's "held-out" means sibling pages within the same site, and its own Table 4 documents that cross-domain transfer "drops performance substantially."** True cross-site generalization is this project's explicit target and their documented weak point. (d) Their CommonCrawl labels are noisy LLM pseudo-labels (best baseline ~40% F1); ours are exact.
- **[Program-as-Weights (Deng et al., arXiv 2607.02512)](https://arxiv.org/abs/2607.02512)** — framing anchor, not a benchmark. Compiles an English spec into LoRA adapter weights via a 4B hypernetwork in one forward pass. Reports 73.78% exact-match on FuzzyBench with a Qwen3-0.6B interpreter, beating direct-prompted Qwen3-32B (68.70%) at ~1/50th inference memory, ~430MB GGUF base plus a 23MB adapter per program at 30 tok/s on an M3. This project is the **code-compilation sibling**: same problem class, different compilation target. The contrast is honest and favorable in one direction — PAW's artifact is still a neural net requiring an inference runtime; a Python function has zero inference cost and can be read and audited by a human. **Artifacts are unreleased**, so PAW is cited and contrasted, never benchmarked against.
- **[SemiBench (facebookresearch/SemiBench)](https://github.com/facebookresearch/SemiBench)** — public, CC BY-NC. 139 real Common Crawl sites with triple annotations (83 single pages, 46 groups of 3, 10 groups of 13). **This is the benchmark SCRIBES reports on**, making a direct comparison to the closest prior art possible.
- **SWDE** — ~124k pages, 80 sites, 8 verticals, 32 human-labeled attributes, with an established "train on k seed sites, test on the remaining unseen sites" protocol built precisely for cross-site generalization. Available via [Academic Torrents](https://academictorrents.com/details/411576c7e80787e4b40452360f5f24acba9b5159) and as a task in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/swde/README.md).
- **Reward-hacking literature** (arXiv 2604.15149, "LLMs Gaming Verifiers") — motivates the leakage analysis and the build-time negative capability tests below.
- **Prior-project literature already in `RESULTS.md`** — Search-R1, R1-Searcher, HiPRAG, RAG-RL, DeepSeek-R1. Two conclusions carry forward directly: outcome-only reward is the right starting point (strategies emerge without process supervision), and curriculum learning (RAG-RL) is the standard mitigation for a cold start.

## What is NOT proven, stated honestly

- Whether a 7B LoRA model can write extraction programs that generalize across *sites* is untested. SCRIBES, at 14B–70B with full finetuning, reports modest absolute numbers (33.2% F1 on SemiBench) and explicitly weak cross-domain transfer. A negative result here is plausible and would replicate their documented finding at smaller scale.
- Whether the efficiency term meaningfully shapes behavior at small λ, or is simply noise, is an open empirical question.
- This project's contribution is **incremental relative to SCRIBES**, not novel. It stress-tests a documented weakness at ~1/100th the compute. That framing is stated plainly in the write-up rather than obscured.

## Why this is not redundant with existing tools

Kadoa, Firecrawl, Bright Data's Scraper Studio, and Browse AI all generate or self-heal extraction code today. They do so by calling a frontier model, at inference time, per site and often per page. None of them train a policy, and none report cross-site generalization as a measured quantity.

The distinction that matters is not "we also generate scrapers." It is that this project treats **generalization as the trained objective** — the reward pays only for programs that survive on unseen sites — and produces a small, local, auditable policy rather than an API dependency.

There is also a case those products structurally cannot serve: **auth-walled or confidential data.** Extraction here happens inside a local Python function; nothing leaves the machine. For internal admin panels, paid data subscriptions, or anything under NDA, sending page contents to a third-party API may be flatly disallowed.

## The task definition

Given `(natural_language_request, page_html)` where the page comes from a **training site**, the agent explores by emitting Python code into a persistent sandbox, observing real execution output, and iterating. It terminates by emitting:

```
SUBMIT:
def parse(html: str) -> list[dict]:
    ...
```

That function is then executed — in a separate, clean, instrumented process — against pages from **held-out sites the agent never explored**, and its output compared to ground truth.

Primary domain: **news / article pages** (schema.org `NewsArticle` / `Article`), where JSON-LD adoption among major publishers is strong. Secondary benchmarks (SemiBench, SWDE) span other verticals, which yields a bonus generalization claim: *trained on news, evaluated cross-vertical.*

## Reward design

```
reward = pass_rate × (1 − λ · normalized_cost)
```

- `pass_rate` — field-level F1 across every record × field, averaged over all held-out pages. A crash, timeout, or empty return scores 0 for that page. **Dense by construction**, which is the direct fix for DocTracerRL's sparse-reward failure.
- `normalized_cost` — runtime and code size of the **submitted program**, normalized within the GRPO group.
- `λ ∈ [0.1, 0.2]` — deliberately small.

**Three decisions here are load-bearing and were reached against known failure modes.**

**1. Cost measures the produced program, not the agent's exploration.** `RESULTS.md:124` records that an efficiency bonus "incentivizes skipping retrieval to guess," and `RESULTS.md:159` documents that it empirically inverted the policy ranking before being removed. Charging for exploration taxes the exact behavior this project exists to grow. Program cost is an outcome property; exploration cost is process supervision in disguise.

**2. λ must be small, because multiplication alone is insufficient.** Multiplicative gating guarantees correctness dominates only when correctness is binary. `pass_rate` is continuous, so without a small λ the arithmetic betrays you: a program passing 30% of tests at zero cost scores 0.30, beating a program passing 90% at high cost scoring 0.27. Bounding the cost term to a 10–20% swing makes it impossible for efficiency to outrank a real correctness gap.

**3. Cost must be measured in a separate, clean process.** Measuring the submitted program's runtime inside the exploration interpreter is invalid: warm imports, an already-parsed soup object, and cached state all contaminate the timing — and the agent can *deliberately* precompute during exploration to make its program appear fast. This is why the scoring executor below is not optional scope.

## Data pipeline

No GPU required; this is CPU and storage work.

1. **Pull** pages from Common Crawl carrying schema.org JSON-LD, filtered to news/article publishers for the primary set.
2. **Split each page in two.** The JSON-LD block becomes the **answer key**. The remainder becomes the **agent input**.
3. **Strip leakage vectors from the agent input:** the JSON-LD block itself, plus `<link rel="canonical">` and `og:url`. Pages routinely contain their own URL, which means the address of the answer key would otherwise be sitting inside the input.
4. **Split by site, never by page.** Held-out sites are never explored, only scored against.
5. **Curriculum tiers** by how far held-out sites drift from training sites — same tag structure with different class names, then different structure, then different value encoding (text price → data attribute, numeric rating → star images). Training begins at the easiest tier so early rollouts have reward variance rather than the all-zeros cold start that killed the 1.5B run.

Ground truth is exact, requires no human labeling, and scales with Common Crawl.

**Manual integrity check before full-scale generation** (carried over from the triage spec): hand-inspect a sample of stripped pages to confirm the answer key is genuinely unrecoverable from what the agent sees, and that the fields are actually present in the visible markup rather than only in the stripped block.

## Sandbox and infrastructure

Estimated 4–5 days. Three components, all load-bearing.

**1. Persistent interpreter for exploration (replaces the cumulative-script design).** The current `PersistentREPL` persists state by re-running a growing script from scratch every step (`repl.py:125`). For HTML parsing this is quadratic — the page is re-parsed on every step — and side effects silently repeat. Required changes:

- A genuinely persistent interpreter process, not re-execution.
- `setsid()` in the spawn hook alongside `setgid`/`setuid`, so the interpreter is a process-group leader and a timeout can reap the **entire tree** via `killpg`. This must be decided at spawn; there is no retrofit, and without it orphaned children hold the stdout pipe open and contaminate later reads.
- An **exit-marker protocol** for completion detection (a persistent pipe has no natural completion signal): append a sentinel echo after each command, poll until it appears, slice output before it.
- **Timeout as a state transition the agent is told about.** On timeout: kill the group, start a fresh session, return the partial stdout *plus an explicit statement that the session was restarted and prior variables are gone.* This is a new failure mode introduced by moving off cumulative re-execution — today state survives a timeout by accident; with a real interpreter it does not.

**2. Separate instrumented scoring executor.** Fresh process per candidate program per held-out page. No exploration state, no warm caches, timed and sized. Serves both cost measurement and correctness scoring.

**3. Isolation.** Threat model is dominated by one hack: **the agent refetching the original page to read the stripped answer key.**

- **Docker becomes mandatory for training.** `verifiers_env.py:78` currently defaults `use_docker=False`, meaning the env falls back to `LocalREPL` unless a caller overrides it — today safe only by accident, since `train_grpo_custom.py:390` happens to pass `None`. Delete the local training path rather than maintain two modes that drift.
- **`--network none`**, already present at `repl.py:94`. Sufficient here and strictly simpler than uid-scoped iptables, because unlike pm_env our harness runs *outside* the container — only agent code runs inside, so nothing in the sandbox needs egress.
- **Scrub the environment.** `repl.py:227` does `env = os.environ.copy()`, handing agent-written code the entire parent environment including `ANTHROPIC_API_KEY`, on a training pod, to arbitrary model-generated code. Pass an explicit allowlist instead.
- **Non-root uid**, and `chmod 0700` on anything the agent must not read.
- **Build-time negative capability tests.** A build layer asserting the agent uid *cannot* resolve DNS and *cannot* read answer-key data, so a leaky image cannot be built. Asserting the absence of a capability, in the artifact itself, is the right tool for this specific threat.

**IP note:** these are standard Unix techniques and are free to use. Implementations from a prior employer's private codebase are not — reimplement from the pattern, do not copy files into a public repo.

**Reuse unchanged:** the `vf.MultiTurnEnv` structure in `verifiers_env.py`, TRL's `GRPOTrainer`, the RunPod workflow, and the pinned `requirements-pod.txt`. Submission parsing already happens in the harness before REPL execution (`verifiers_env.py:139`), so the reward channel is already unreachable from inside the sandbox — keep that property.

## Training

- **Base model:** Qwen2.5-Coder-7B-Instruct, LoRA. The task is fundamentally code generation, which argues for the Coder variant; Qwen2.5-7B-Instruct is the validated fallback, having already been shown to emit valid Python in this environment.
- **Warm-start:** RAFT / rejection sampling. Sample many candidate programs per task at temperature, run each through the verifier, keep the passers as SFT data. `assistant_only_loss=True`, carried over. Target is thousands of auto-labeled examples, not dozens.
- **GRPO:** TRL's `GRPOTrainer`, group size 8, fixed-length reward normalization, skip the gradient step when a group's rewards are uniform. Curriculum from easiest to hardest drift tier.
- **Mandatory gate:** a 15–20 step, ≤$5 diagnostic run with an explicit go/no-go decision recorded before committing further budget.
- **Budget:** roughly $10–40 on an RTX 4090 (~$0.34/hr) or A6000 (~$0.49/hr) community-cloud pod, billed per second. Comfortable headroom inside $100. The genuine unknown is per-episode token cost, since HTML observations are larger than document snippets — which is exactly what the diagnostic gate exists to measure.

**DPO was explicitly considered and rejected.** It is dominated on both sides: GRPO is strictly better for signal because it is on-policy and improves its own data as the policy improves, while RAFT is strictly simpler if the goal is cheap stability (no reference model, no beta). DPO is also formulated over a single prompt-response pair, so applying it to a multi-turn exploration trajectory would give credit assignment *coarser* than the trajectory-level F1 that already failed once. It would only become correct if the verifier were lost and scoring became subjective, which this design guarantees will not happen.

## Evaluation

Held-out sites, never explored during training.

**Policies compared:** untrained Qwen2.5-Coder-7B base (required — isolates the training contribution), RAFT-only checkpoint, GRPO checkpoint, and a naive one-shot baseline (write a parser from the page without exploring). A frontier model writing a parser one-shot is a useful contextual reference, not the primary claim.

**Primary metric:** field-level F1 on held-out sites, reported per curriculum tier so cross-site drift is visible rather than averaged away.

**Secondary:** SemiBench, for direct comparison against SCRIBES's published numbers. *Wrinkle to handle explicitly:* SemiBench annotates subject-predicate-object triples while this design outputs flat field-value records, so a small adaptation layer is required. SWDE optional as a classic reference point.

**Cost-crossover analysis:** measured LLM cost per page versus compiled-parser cost per page, with the break-even page count stated. Estimated in the low tens of pages; the measured number belongs in the write-up.

**Leakage audit, required before trusting any result:** confirm no submitted program achieves high scores via network access, embedded URLs, or any path back to the answer key. Verify the build-time capability tests are actually running in CI rather than silently skipped.

## Success criteria

- **Must-have:** the GRPO checkpoint beats the untrained base on held-out cross-site F1, with the delta reported honestly either way.
- **Stretch:** GRPO beats RAFT-only, evidencing that on-policy optimization adds something rejection-sampled imitation does not.
- **Failure is reportable:** if cross-site transfer stays flat, that replicates SCRIBES's own documented cross-domain weakness at 1/100th the compute, and is written up as a replication rather than an unexplained null result.

## Deliverables

1. Environment, sandbox, data pipeline, training, and eval code — public repo, everything released. **PAW's artifacts are unreleased; full reproducibility is a real, checkable property this project has and its framing anchor does not.**
2. Results write-up: policy comparison table, per-tier cross-site breakdown, cost-crossover chart, and an explicit statement of what was and was not beaten.
3. **A usable CLI:** paste a URL and a request in English, get a parser file back. A thing people can run beats a thing people read, and it costs roughly a day on top of what is already scoped.

## Demo and write-up assets

Logging must be in place **before** the full GRPO run and final eval; transcripts cannot be reconstructed afterward.

- **The redesign test is the lead asset.** The trained model's parser working on a site it never saw, next to the untrained model's parser returning nothing on the same page. This lands with any audience and needs no explanation.
- **Playwright visualization** (demo only, explicitly not in the training loop): render the real page and highlight the elements the submitted parser extracted. Confirmed as tooling, not environment — a browser-driven artifact would be expensive per page and would undercut the cheap-reusable-artifact argument.
- **Exploration transcript recording:** the trained agent poking at a page — printing tag counts, trying a selector, noticing it got 3 results where there should be 12, adjusting, submitting.
- **Fixed test pages frozen before training starts**, re-run through every checkpoint (base, RAFT, GRPO) for a before/after comparison across training stages.
- **Insurance note:** build these early. If the F1 delta turns out modest, the redesign demo carries the project on its own — a parser surviving a layout change is visibly impressive regardless of the size of the improvement.

## Schedule (5 weeks)

1. **Sandbox rebuild** — persistent interpreter with `setsid` + exit-marker + timeout-as-state-transition, separate instrumented scoring executor, isolation hardening, build-time negative capability tests passing in CI. *End of week 1.*
2. **Data pipeline** — Common Crawl JSON-LD extraction, leakage stripping, site-level splits, curriculum tiers, manual integrity check passed. *End of week 2.*
3. **Baselines** — untrained base and one-shot no-exploration policies scored on held-out sites. *End of week 2.*
4. **RAFT warm-start** — thousands of verifier-filtered programs, checkpoint beats the untrained base. *Mid week 3.*
5. **Cheap GRPO diagnostic** — 15–20 steps, ≤$5, explicit go/no-go recorded. *End of week 3.*
6. **Full GRPO run** — with curriculum, inside budget. *Week 4.*
7. **Final eval** — policy table, per-tier breakdown, SemiBench comparison, leakage audit. *End of week 4.*
8. **Ship** — CLI, demo assets, write-up. Project named. *Week 5.*

## Risks

- **Capability ceiling.** Writing a parser that generalizes across sites is harder than the tasks a 7B model reliably handles, and SCRIBES's modest absolute numbers at much larger scale are a warning. Mitigated by the curriculum, the dense reward, and the diagnostic gate — not eliminated. This is the project's single largest risk.
- **Cold start.** If every early rollout scores near zero, GRPO has no gradient — the exact 1.5B failure. Mitigated by starting at the easiest curriculum tier and by dense field-level partial credit.
- **Answer-key leakage.** Highest-payoff reward hack available. Mitigated by network isolation, URL stripping, build-time capability tests, and an explicit pre-result audit.
- **λ turns out to be noise.** Possible; report it honestly as a negative finding about the cost term rather than quietly dropping it.
- **Scope.** This is a fuller build than the triage router — new sandbox, new data pipeline, new reward, new eval. The sandbox is capped at 4–5 days deliberately; it reads as infrastructure in a write-up and must not expand into a co-equal deliverable.
- **SemiBench format mismatch.** Triples versus flat records requires an adaptation layer. Named here so it is scoped rather than discovered late.
