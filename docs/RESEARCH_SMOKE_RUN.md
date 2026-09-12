# First three-question Qwen run

Prepared September 12, 2026. This is an inference smoke test; no model weights are updated.

## Purpose and decision rule

Hypothesis: Qwen3-8B can use the existing research tools across turns and submit an answer whose
evidence can be inspected. This first run cannot establish superiority over Qwen2.5 or an
improvement from training.

Run `research_01` (unsupported claims), `research_04` (small-model search training), and
`research_11` (latest-research coverage). Use the frozen six-paper snapshot, greedy non-thinking
decoding, seed 42, at most 10 actions, and at most 1,800 generated tokens per action.

Expected signal: successful tool use, traceable evidence, and disclosure that the historical
starter collection cannot establish the latest research. Stop the batch on a server error or
failure to submit. Inspect tool errors and every claim before expanding to 20 prompts. A
syntactically valid submission or an exact quotation is not proof of answer quality.

## Actual first-question outcome

The supervised smoke test used the pinned Qwen3-8B revision above. After the first Secure RTX 4090
pod was stopped, that host had no capacity to restart it, so the corrected retry ran on a Secure
NVIDIA A40 pod at the RunPod-listed rate of $0.49/hour. The replacement pod was stopped immediately
after the saved result reported `EXITED`; neither pod was deleted.

The corrected `research_01` run reached a valid submission at step 10. It searched once and read
three passages, and the runner successfully resolved the submitted document offsets into exact
snapshot text. This validates the source-span mechanism. The answer did not satisfy the requested
two-approach comparison: it preserved only one claim, repeated malformed actions after a syntax
failure, and recommended testing a mechanism after reporting that its paper found no improvement.
Semantic support and project usefulness remain human-review judgments.

Artifacts are under `out/research/gpu-smoke-20260912/`; the final corrected files are
`source-span-retry-research_01.json` and `.md`. This result supports continuing the benchmark
pipeline, not a claim that Qwen is already a good research agent.

This historical run used `research-evidence-v1`, where Qwen wrote Python actions. The current
`research-tools-v2` runner accepts only validated JSON research actions and executes the four
read-only tools in the application. Replaying the current pilot therefore does not require Docker.

## Resources and current state

- Code is committed locally at `3eff49e` (`pivot to ai research`).
- The paper snapshot exists at `out/research/starter-2026-09-12`.
- RunPod reported no current pods during preparation.
- RTX 4090 24GB was listed at $0.34/hour in Community Cloud, with low stock in EU-CZ-1.
  This is a catalog quote, not a guaranteed launch price, and excludes storage.
- RTX A6000 was out of stock during the check.
- The RunPod connector does not expose the account's credit balance through its current tools.
  A missing-balance response must not be interpreted as zero credit.
- The local `rlm-sandbox` image is built. A real Docker check passed for the six-paper mount,
  search, state persistence across actions, and exact passage retrieval.
  Image ID: `sha256:921a03f01c2a4c691e0bf54b5901a9e1a235ea492a95d02ea8c7a1f85496dae6`.
- The user subsequently asked to launch the test. Proceeding under the proposed $2 budget,
  a Secure Cloud RTX 4090 was allocated at $0.74/hour after Community Cloud allocation failed.
  See `out/research/gpu-smoke-20260912/provisioning.json` for the allocation and timestamps.

Use a single GPU. Download/startup time counts as rental time. Reserve up to an hour for the
initial supervised setup and test; this is a planning allowance, not a runtime guarantee. Check
the actual hourly rate after allocation, collect the output, and stop the pod promptly. Stopping
the model process alone does not stop RunPod billing; retained disk/volume storage may still cost
money after the GPU is stopped. Do not describe a tmux session as an automatic budget cap.

## Concrete deployment configuration

Use the RunPod-recommended general-purpose image:
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, one RTX 4090, and a host advertising CUDA 12.8
or newer. A 60GB container disk and no separate persistent volume are sufficient starting
allocations for this disposable inference session; check free space before downloading weights.
Authorize an SSH public key and expose SSH only. Bind the model server to loopback and reach it
through an SSH tunnel.

Start with this pinned Qwen3-8B revision:
`b968826d9c46dd6066d109eabc6255188de91218`, resolved from
[the public Hugging Face metadata](https://huggingface.co/api/models/Qwen/Qwen3-8B).

On the GPU pod, install a separate serving environment and capture the resolved dependencies:

```bash
python3 -m venv /workspace/research-serving
/workspace/research-serving/bin/pip install 'vllm==0.10.2' \
  --extra-index-url https://download.pytorch.org/whl/cu128
/workspace/research-serving/bin/pip freeze > /workspace/research-serving.freeze.txt

/workspace/research-serving/bin/vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --generation-config vllm
```

This is a candidate configuration to validate on the GPU, not an already-tested memory fit.
Use the [vLLM 0.10.2 installation reference](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html).
Do not install these serving dependencies over the legacy GRPO training environment. Record
`nvidia-smi`, the server command, model revision, and package freeze with the run artifacts.

Launch the server in a pod-side tmux session if tmux is available. This keeps the server alive
through an SSH disconnect, but the local agent runner below still requires the laptop to remain
awake. The small test is supervised; unattended remote orchestration is not implemented here.

## Run the local agent against the GPU server

Forward the pod's loopback server to the laptop:

```bash
ssh -N -L 8000:127.0.0.1:8000 -p POD_SSH_PORT root@POD_HOST
```

In a separate local shell, from the repository root:

```bash
export RESEARCH_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export RESEARCH_SERVER_HARDWARE='1x RTX 4090 24GB; BF16; vLLM 0.10.2; context 16384; eager'
bash scripts/run_research_smoke.sh
```

Replace the hardware description with the actual serving configuration if it differs. The runner
checks the snapshot before making model calls. Model-generated JSON is validated against an
allowlist, and the local application executes only bounded read-only research tools; the GPU pod
only serves inference. The runner stops on a non-submitted episode and preserves its error artifact.

Review each answer and its full trace. Record unsupported claims, missing evidence, practical
usefulness, date/coverage disclosure, and tool failures. Then save the outputs and stop the pod
through RunPod. Verify its stopped/deleted state in the connector; a dead SSH connection is not
proof that billing has stopped.
