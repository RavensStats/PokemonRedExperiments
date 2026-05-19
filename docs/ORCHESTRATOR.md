**Orchestrator Overview**

This document summarizes how the Python round-robin orchestrator works for running multi-config, multi-seed sweeps.

**Location**: [scripts/run_compare_four_configs_round_robin_v2.py](scripts/run_compare_four_configs_round_robin_v2.py)

**Purpose**:
- **Goal**: Run a set of configurations across multiple random seeds in a resumable, cross-platform way and perform post-evaluations after training completes.

**Key Behaviors**:
- **Round-robin scheduling**: Cycles through configs × seeds, launching one training job at a time to avoid resource contention.
- **Resumability**: Each run writes an `exit_code.txt` and `run_metadata.json` into its `run_dir`. The orchestrator skips runs that have a completed `exit_code.txt` (so re-running with `--resume` continues where it left off).
- **Atomic run setup**: The `run_dir` is created before writing `run_metadata.json` to avoid race conditions or missing directories.
- **Trainer invocation**: Calls the trainer directly (`v2/baseline_fast_v2.py`) with the configured args (device, num-cpu, total-timesteps, discovered-events flags, etc.).
- **No empty promoted-path**: The orchestrator avoids passing an empty `--discovered-events-promoted-path` string to the trainer (this was a common argparse issue).

**Run lifecycle**:
- Create `run_dir` (sweeps/<sweep_name>/<config>/run_<NNN>_seed<S>)
- Write `run_metadata.json` (index, seed, config, full command, train args)
- Launch trainer process and stream logs to `stdout.log` (optionally gzipped)
- Trainer writes TensorBoard events under `poke_ppo_1/`
- On process exit the orchestrator writes `exit_code.txt` (numeric return code). If rc==0 then run considered successful.

**Failure semantics**:
- Native crashes (e.g. exit code 0xC0000409) are written to `exit_code.txt` and stop the orchestrator run loop. These typically indicate a native/C-extension or driver issue (CUDA, PyBoy, etc.).
- The orchestrator is conservative: it stops and reports a failure so the user can inspect logs before continuing.

**Post-evaluations**:
- After all training runs complete successfully, the orchestrator invokes the evaluation scripts in order:
  - `scripts/eval_input_jitter_best_v2.py` — runs input-jitter robustness tests on the best checkpoint per config
  - `scripts/eval_perception_noise_best_v2.py` — runs perception-noise robustness tests on the best checkpoint per config
- Eval scripts sanitize JSON payloads (convert Path objects) and require `summary.csv` to exist (the orchestrator ensures training runs produce summaries before evals).

**Resume / Recovering a failed run**:
- To resume the sweep (skip runs already finished):

```powershell
.\.venv\Scripts\python.exe scripts\run_compare_four_configs_round_robin_v2.py --resume --runs 20 --seed-start 0 --num-cpu 4 --total-timesteps 330000 --device auto --no-stream
```

- To re-run a single failing job as a short smoke test (quick reproduce, capture full stdout/stderr):

```powershell
.\.venv\Scripts\python.exe v2\baseline_fast_v2.py --run-dir sweeps\sweep_20260517_175436\dynamic_no_promoted_in_rank\run_004_seed4 --seed 4 --total-timesteps 1000 --device cpu --no-stream
```

**Useful files per run**:
- `run_metadata.json` — canonical record of the command and args used for the run
- `stdout.log` / `stdout.log.gz` — captured trainer output
- `exit_code.txt` — numeric return code (used to decide skip/resume)
- `poke_ppo_1/` — TensorBoard event files written by the trainer

**Notes & Recommendations**:
- If you see native exit codes (e.g. `0xC0000409`), try re-running the job on CPU to isolate GPU/driver issues.
- Keep `exit_code.txt` around as the authoritative marker; manually deleting it signals the orchestrator to attempt the run again.
- For debugging, run the trainer directly with `--total-timesteps 1000` and `--no-stream` to get faster, clearer logs.

If you want, I can (a) add a short troubleshooting checklist to this file, (b) add a small wrapper to rerun failed jobs automatically, or (c) update the orchestrator to continue on failure instead of stopping. Which would you prefer?
