---
name: rtx5090-connect
description: Connect from the DGX Spark to the remote RTX 5090 Ubuntu workstation (ss-Fusion-75) and dispatch, monitor, or retrieve GPU training/fine-tuning jobs there over SSH + Docker. Use this whenever the user asks to run, submit, launch, check on, or pull results from a job on "the 5090", "the RTX 5090 box", "the workstation", "fusion75"/"ss-Fusion-75", or generally wants to use a remote/second GPU for training or fine-tuning instead of (or in addition to) the DGX Spark's own onboard GPU. Also use this to sanity-check that the remote GPU pipeline is healthy before trusting it with a long run, even if the user doesn't explicitly name the machine -- if they mention pretraining or fine-tuning happening on "the other machine" or "the workstation", this is almost certainly it.
---

# RTX 5090 remote GPU connect

This skill is a callable replacement for manually re-deriving SSH/Docker
connection details every time a job needs to go from the DGX Spark to the
RTX 5090 workstation. It wraps already-verified infrastructure (built and
tested 2026-07-08) so you don't have to re-discover it from scratch, and it
enforces the one gotcha that silently produces wrong results if skipped.

## Why this exists / the one thing to never skip

**The DGX Spark has its own onboard Blackwell GPU (GB10 Superchip).** Any GPU
command run in a *local* terminal on the Spark shows the Spark's own GPU, not
the target. A command can succeed, return a plausible-looking GPU name, and
still be running against entirely the wrong hardware. Every script here
calls `verify_target_gpu` (in `scripts/_common.sh`) before doing anything,
which SSHes into the target and checks the returned GPU name literally
contains `"RTX 5090"`. Never bypass this check, and never assume success from
a local nvidia-smi call.

## Target environment (what you're connecting to)

- Host: `ss@192.168.0.204` (LAN, DHCP -- can drift; see Troubleshooting).
- Auth: SSH key only, password auth disabled on the target. Your key must
  already be in the target's `~/.ssh/authorized_keys`.
- GPU stack: native Ubuntu 26.04, NVIDIA driver (open kernel module,
  persistence mode on) -- no WSL, no VM, no Windows anywhere in the path.
- Jobs run inside a persistent Docker container (`docker compose`, service
  `train`, image `rtx5090-train:latest` -- NGC PyTorch base with
  transformers/accelerate/peft/trl/datasets/bitsandbytes/deepspeed/wandb/
  jupyterlab preinstalled). The container stays up between jobs (`exec`
  into it, don't `docker run` a fresh one per job) so the HuggingFace model
  cache and any warm state persist.
- Two shared dirs bind-mounted into the container: `~/fusion-project` on the
  target (persistent -- use this for real jobs) and
  `~/rtx5090setup/docker/workspace` (scratch/throwaway).
- A full manual runbook lives on the target at
  `~/rtx5090setup/dgx-spark-to-rtx5090-job-instructions.md` for anything this
  skill doesn't cover -- read it over SSH (`ssh <host> "cat <path>"`) rather
  than guessing.

## Workflow

Run these from wherever this skill directory lives (they `cd` to their own
location to find `_common.sh`, so invoke by path, e.g.
`bash scripts/connect_check.sh` from inside the skill dir, or the full path
from anywhere else).

### 1. Verify the target is reachable and it's really the 5090

```bash
bash scripts/connect_check.sh
```
Confirms SSH connectivity, that the GPU is genuinely an RTX 5090, that Docker
is active, and that the `train` container is running (starts it if not).
Run this first, always -- it's cheap and it's what catches the
wrong-machine mistake before it wastes a training run.

### 2. Run the smoke test before trusting a long job to the box

```bash
bash scripts/smoke_test.sh 50
```
Trains a small transformer for N steps (default 50, takes seconds) as a
stability/health check. Worth running any time something feels off -- a
prior job hung, the connection dropped mid-run, or it's just been a while
since the box was last used. A clean exit here doesn't guarantee a multi-hour
run will be flawless, but a failure here means don't even try.

### 3. Submit a job

```bash
bash scripts/submit_job.sh <local_job_dir> <remote_job_name> [entrypoint.py]
```
Copies `<local_job_dir>` to `~/fusion-project/<remote_job_name>` on the
target and launches `<entrypoint.py>` (default `train.py`) detached inside
the running container, with output redirected to `train.log` in the job
directory. Prints the first lines of that log once it appears so you know
the job actually started (imports/model downloads can delay the first log
line -- a WARN here isn't necessarily a failure, just check again with
`status.sh` shortly after).

### 4. Monitor it

```bash
bash scripts/status.sh <remote_job_name>
```
Tails the job's log, reports live GPU utilization/memory/temperature/power,
and lists whether a python process is still alive inside the container. Use
GPU utilization as the tell for "is this actually training" vs. "hung but
the process is still technically alive" -- a job stuck at 0% GPU util for
more than a few minutes after startup is a hang, not slow progress.

### 5. Retrieve results

```bash
bash scripts/fetch_results.sh <remote_job_name> <local_dest_dir> [remote_subpath]
```
Pulls `checkpoints/` (default) or any other subpath back to the local
machine via rsync.

### 6. Clean up

```bash
bash scripts/cleanup.sh <remote_job_name>
```
Deletes the job's directory from the target. Asks for confirmation
interactively; set `CONFIRM=yes` to skip the prompt in non-interactive/agent
contexts (only after results are already fetched -- this is destructive and
not reversible).

## Overriding the target host

If `192.168.0.204` no longer resolves (DHCP reassignment is the most likely
cause), don't guess -- ask the user for the current IP/hostname, or check
`arp -a` / the router's DHCP lease table if you have access, then pass it
explicitly rather than editing the scripts:

```bash
RTX5090_HOST=ss@<new-ip> bash scripts/connect_check.sh
```

Every script in `scripts/` honors `RTX5090_HOST` as an environment variable
or trailing positional argument (see each script's usage comment).

## Troubleshooting

`connect_check.sh` failing is the entry point for diagnosing anything here
-- it prints specific guidance for each failure mode (unreachable host, key
not authorized, GPU mismatch). For anything deeper, the full runbook on the
target has more context than is worth duplicating here:

```bash
ssh ss@192.168.0.204 "cat ~/rtx5090setup/dgx-spark-to-rtx5090-job-instructions.md"
```

## Notes for non-Claude-Code agents (e.g. Cursor)

Everything above is plain bash/ssh/rsync/docker -- there's nothing
Claude-Code-specific in the mechanics. In Cursor (or any agent without a
native "skill" loading mechanism), just point the agent at this file as
context/a rule, or read it directly and run the scripts by path; the
scripts themselves have no dependency on being invoked by Claude Code.
