# RL_Logic.md — How This Repo Works (Pokemon Red RL)

This repo trains reinforcement-learning agents to play **Pokémon Red** by running the game in an emulator (PyBoy), exposing a Gymnasium environment, and training PPO (Stable-Baselines3) on observations that combine **pixels + game-state features**. The repo has two main “generations” of code:

- **`v2/` (recommended / actively used):** coordinate-based exploration, dict observations, cleaner reward shaping.
- **`baselines/` (older):** frame-KNN novelty exploration, monolithic image observation that includes “memory overlays”.

Below is an end-to-end explanation of the modules, why they exist, and how data flows between them.

---

## 1) High-level architecture

At a high level, you have:

- **Training entrypoint** (creates env(s), chooses PPO params, sets callbacks)
- **Environment** (emulator + observation + reward)
- **Callbacks/logging** (TensorBoard scalars/histograms/images, checkpoints)
- **Experiment orchestration** (sweeps, iterative “tournament” training)
- **Optional streaming/visualization** (WebSocket coordinate broadcast + notebooks/scripts)

### Module flow diagram

```mermaid
flowchart LR
  subgraph Orchestration
    A1[scripts/run_sweep_v2.py\nN independent runs] -->|launches subprocess| B
    A2[scripts/run_iterative_sweep_v2.py\nstage->rank->resume] -->|launches subprocess| B
  end

  subgraph TrainingProcess[One training process (one run folder)]
    B[v2/baseline_fast_v2.py\nSB3 PPO trainer] --> C[SubprocVecEnv\n(num-cpu env workers)]
    B --> D[CheckpointCallback\nmodel checkpoints]
    B --> E[TensorboardCallback\nenv_stats + maps]
  end

  subgraph EnvWorkers[Env subprocesses (inside SubprocVecEnv)]
    C --> F1[make_env(rank)]
    F1 --> G[RedGymEnv\nv2/red_gym_env_v2.py]
    F1 --> H[StreamWrapper (optional)\nv2/stream_agent_wrapper.py]
    H --> G
  end

  subgraph Emulator
    G --> I[PyBoy emulator\nloads .gb + .state]
    G --> J[Reads RAM addresses\ncoords/events/badges/HP]
  end

  subgraph DataFiles
    K1[v2/events.json\nflag name map] --> G
    K2[v2/map_data.json\nmap rectangles] --> L[v2/global_map.py\nlocal_to_global + mask]
    L --> G
  end

  E --> M[TensorBoard event files\nevents.out.tfevents.*]
  E --> N[trajectory images\nexplore maps]
  D --> O[poke_*_steps.zip\ncheckpoints]

  subgraph Analysis
    P1[scripts/summarize_sweep_v2.py\nrank runs] --> M
    P2[scripts/plot_sweep_v2.py\ncurves + AUC] --> M
  end
```

---

## 2) “Run” vs “Env worker” (important mental model)

There are two nested layers of parallelism:

1) **Run (seed-level):** one OS process running `v2/baseline_fast_v2.py` with a single output directory (checkpoints, TensorBoard logs, stdout).

2) **Env worker (CPU-level):** inside that one run, `SubprocVecEnv` starts `--num-cpu` subprocesses, each running its own emulator instance, generating experience in parallel for the same PPO model update.

This is why a sweep can be “many runs” (many seeds / processes) while each run itself can also have “many envs”.

---

## 3) Orchestration scripts (multi-run automation)

### 3.1 `scripts/run_sweep_v2.py`

Purpose: run *many independent training runs* (different seeds), optionally with a wall-clock budget.

Key ideas:

- Creates a sweep folder: `sweeps/sweep_<timestamp>.../`
- Creates run folders: `run_000_seed0/`, `run_001_seed1/`, ...
- Launches `v2/baseline_fast_v2.py` as a subprocess per run
- Captures stdout/stderr into `stdout.log.gz` (lossless) by default

The subprocess command is assembled in `_build_train_cmd(...)` and forwards most CLI args to the V2 trainer.

### 3.2 `scripts/run_iterative_sweep_v2.py` (added)

Purpose: implement a “tournament” loop:

- Stage 0: run N candidates for ~2 hours
- Summarize + rank
- Pick best checkpoint
- Stage 1: resume training from that checkpoint for another ~2 hours
- Repeat

This uses the existing summary logic (`scripts/summarize_sweep_v2.py`) and the built-in checkpoint naming (`poke_<steps>_steps.zip`).

What it selects:

- It ranks by final values of scalar metrics (default ordering):
  1. badges
  2. events_completed
  3. explore_pct
  4. max_level

Those metrics come from TensorBoard logs written by `TensorboardCallback` (see below).

---

## 4) Training entrypoint (V2)

### 4.1 `v2/baseline_fast_v2.py`

This is the main training script.

What it does:

1) Parses CLI args (run dir, seed, `num-cpu`, emulator params, PPO params).
2) Builds an `env_config` dict that is passed to each environment instance.
3) Creates `SubprocVecEnv([make_env(rank, env_config, seed=seed) ...])`.
4) Creates callbacks:
   - `CheckpointCallback(save_freq=ep_length//2, save_path=sess_path, name_prefix="poke")`
   - `TensorboardCallback(sess_path)`
   - Optional: W&B integration
5) Loads a checkpoint if `--resume` is provided (or via stdin), otherwise creates a new PPO model.
6) Calls `model.learn(total_timesteps=..., callback=CallbackList([...]))`.

Why the design looks like this:

- **`SubprocVecEnv`** is used to parallelize experience collection across multiple emulator instances.
- **PPO** is chosen as a stable, commonly-used on-policy algorithm for high-dimensional / partially observed environments.
- **Callbacks** keep “training logic” separate from “metrics/checkpoint concerns” and allow sweeps to parse results later.

---

## 5) Environment (V2): emulator, observation, action, reward

### 5.1 `v2/red_gym_env_v2.py` — `RedGymEnv`

This is the core Gymnasium environment.

#### Emulator lifecycle

- On `__init__`: creates a PyBoy emulator instance for `PokemonRed.gb` (or whatever `--gb-path` points to).
- On `reset()`: loads a pre-saved `.state` file (e.g. `init.state`) via `pyboy.load_state(...)`.

This is how runs can start from “a good starting point” (skipping intro, optionally already having items).

#### Action space

`spaces.Discrete(len(valid_actions))` where actions map to button presses:

- arrows: up/down/left/right
- A, B, START

The environment presses a button, ticks the emulator, releases it, ticks more:

- `press_step = 8`
- total ticks per action = `action_freq` (default 24)

This creates a “held for a few ticks” behavior, then release, which better matches how real inputs work.

#### Observation space (Dict)

`spaces.Dict(...)` with:

- `screens`: stacked downscaled grayscale frames (shape `(72, 80, 3)`)
- `health`: `[hp_fraction]`
- `level`: Fourier encoding of party level sum
- `badges`: 8 bits from RAM
- `events`: event flag bits from RAM (`(event_flags_end - event_flags_start) * 8`)
- `map`: a local crop of the stitched global explore map around the agent
- `recent_actions`: last K actions (K = frame stack)

Why dict obs:

- It lets the policy consume *heterogeneous* information (pixels + structured RAM-derived signals).
- It avoids forcing “everything” into a single image tensor (which is what `baselines/` does).

#### Per-step flow inside `step(action)`

The sequence in `step()` is roughly:

1) Execute action in emulator (`run_action_on_emulator`).
2) Record per-step stats (`append_agent_stats`).
3) Update action history and screen history.
4) Update exploration memory (`update_seen_coords`, `update_explore_map`).
5) Update shaped rewards (healing, battle damage, level penalty).
6) Compute **delta reward** via `update_reward()`.
7) Decide whether episode is done (`check_if_done`).
8) Return `(obs, reward, terminated=False, truncated=done, info={})`.

#### Exploration representation

V2 uses **coordinate-based exploration**:

- It reads `(x, y, map_id)` from RAM.
- It stores a counter per coordinate string (`seen_coords["x:.. y:.. m:.."] += 1`).
- It maps local coords into a stitched global map with `local_to_global(...)` from `v2/global_map.py`.
- It updates a global `explore_map` image by setting `explore_map[gy, gx] = 255`.

The `map` observation is a crop of `explore_map` around the current location.

#### Reward shaping

The environment maintains a running “progress vector” and returns **the change in total progress** each step:

- `progress_reward = get_game_state_reward()` returns a dict of reward components.
- `total_reward = sum(progress_reward.values())`.
- Step reward = `new_total - old_total`.

Current V2 reward components (from `get_game_state_reward()`):

- `event`: max number of (new) event flags set (scaled)
- `heal`: sum of squared healing increments (only when party size unchanged)
- `badge`: badge bitcount
- `explore`: `len(seen_coords)` (unique coords visited)
- `stuck`: penalty when standing on the same coord too long
- `op_damage`: cumulative opponent HP fraction decrease dealt
- `level_pen`: penalty triggered when a battle starts where enemy level >= player level
- `game_progress`: milestone score from `GAME_MILESTONES` (explicit long-horizon objectives)

Why it’s structured this way (as inferred from code intent):

- **Milestones + badges + events** create a sparse-but-meaningful target signal aligned with “beating the game”.
- **Exploration** prevents local minima (“stand in place”).
- **Stuck penalty** discourages dithering.
- **Battle damage reward** provides dense feedback during fights.
- **Healing reward** encourages visiting Poké Centers / recovery behaviors.

Important detail: `event` and `game_progress` track maxima (`update_max_event_rew`, `update_game_completion_rew`) so reward doesn’t decrease if the agent later moves away.

---

## 6) Global map stitching and exploration %

### 6.1 `v2/global_map.py`

- Loads stitched map rectangles from `v2/map_data.json`.
- Defines `GLOBAL_MAP_SHAPE`.
- Builds `VALID_TILE_MASK`, a boolean mask of “tiles that are part of some known rectangle”.

Why `VALID_TILE_MASK` exists:

- The stitched global map includes padding and empty areas.
- Exploration percentage becomes meaningful when you divide by “tiles that belong to the known world rectangles” rather than the entire padded canvas.

`local_to_global(r, c, map_n)` converts local coordinates to global stitched coordinates and clamps/handles unknown map IDs.

---

## 7) Logging and metrics

### 7.1 Environment-side stats

`RedGymEnv.append_agent_stats(...)` appends a dict each step containing:

- position (x, y, map)
- levels, HP fraction
- coord_count (unique coords)
- explore tiles + explore %
- badge count
- events_completed (event flag count)
- components like `event` and `healr`

These `agent_stats` are later harvested by the callback.

### 7.2 `v2/tensorboard_callback.py` — `TensorboardCallback`

This callback runs inside SB3 and is responsible for writing the metrics that your sweep scripts depend on.

Behavior:

- On training start: creates a `SummaryWriter` in `<run_dir>/histogram/`.
- On each SB3 step: if env 0 reports `check_if_done()` true, it:
  - collects `agent_stats` from all env workers
  - takes the **last** entry from each env’s stats (end-of-episode)
  - logs mean values to TensorBoard under `env_stats/<key>`
  - logs histograms under `env_stats_distribs/<key>`
  - logs max values under `env_stats_max/<key>`
  - logs exploration map images under `trajectory/*`
  - logs a merged JSON of discovered event flags under `trajectory/all_flags`

This is why `scripts/summarize_sweep_v2.py` can rank runs: it reads those scalars from `events.out.tfevents.*`.

### 7.3 Sweep analysis

- `scripts/summarize_sweep_v2.py`:
  - finds each `run_*` directory
  - loads TensorBoard scalars for `env_stats_max/*` (or `env_stats/*`)
  - writes a `summary.csv` with final badge/events/explore/max_level
  - sorts runs descending by those metrics

- `scripts/plot_sweep_v2.py`:
  - loads full curves for those metrics
  - can compute AUC and time-to-threshold goals

---

## 8) Checkpoints and resuming

### 8.1 Checkpoints

`CheckpointCallback(save_freq=ep_length//2, save_path=run_dir, name_prefix="poke")` writes files like:

- `poke_327680_steps.zip`
- `poke_655360_steps.zip`

### 8.2 Resume behavior

`v2/baseline_fast_v2.py` supports:

- `--resume <path>` (with or without `.zip`)

It loads the PPO model and patches a few rollout settings (n_steps / n_envs) to match current CLI.

This is what allows iterative training: “pick checkpoint → resume into next stage”.

---

## 9) Streaming / live map visualization

### 9.1 `v2/stream_agent_wrapper.py` — `StreamWrapper`

This wrapper broadcasts coordinates periodically over a WebSocket:

- reads `(x, y, map)` from emulator RAM
- accumulates coordinate samples
- every `upload_interval` steps (default 300), sends:

```json
{ "metadata": { ... }, "coords": [[x,y,map], ...] }
```

This is optional (`--stream/--no-stream`) and is separate from training correctness.

Why it exists:

- Allows real-time visualization / monitoring of where the agent is exploring.

---

## 10) Baselines (`baselines/`) — what’s different and why

The baseline environment in `baselines/red_gym_env.py` differs in a few key ways:

- Observation is a **single image tensor** that includes:
  - downscaled frames
  - “memory” overlays appended above frames (exploration memory, recent reward memory)
- Exploration reward can be **screen-novelty based**:
  - it flattens the current frame
  - inserts into an HNSW KNN index (`hnswlib`)
  - uses distance-to-nearest-neighbor as novelty signal

Why V2 moved away from that:

- Frame-KNN novelty can be expensive and noisy (new frames from UI flicker, battle animations, etc.).
- Coordinate-based exploration is cheaper, more interpretable, and aligns better with “cover the map”.

The baseline callback `baselines/tensorboard_callback.py` is essentially the same pattern as V2.

---

## 11) Visualization + notebooks

The `visualization/` and various `*.ipynb` notebooks are primarily analysis tools:

- stitching maps
- creating video grids
- plotting progress over time

They generally consume outputs written by runs (TensorBoard event files, explore maps, rollouts) rather than participating in training.

`clip_experiment/` similarly appears to be an orthogonal experiment area (CLIP-based image interactions) rather than core RL training.

---

## 12) Practical workflows

### Single V2 run

- Run trainer:
  - `python v2/baseline_fast_v2.py --run-dir runs/my_run --num-cpu 4 --no-stream`

### Sweep (many seeds)

- `python scripts/run_sweep_v2.py --runs 10 --seed-start 0 --hours 8 --num-cpu 4 --no-stream`
- After completion:
  - `python scripts/summarize_sweep_v2.py sweeps/<your_sweep>`
  - `python scripts/plot_sweep_v2.py sweeps/<your_sweep> --top-k 10`

### Iterative tournament (2h stages)

- `python scripts/run_iterative_sweep_v2.py --stages 6 --stage-hours 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --no-stream`

---

## 13) Where to change things (extension points)

- **Reward shaping / termination:** `v2/red_gym_env_v2.py`
- **Observation design (add/remove channels):** `v2/red_gym_env_v2.py` (`observation_space`, `_get_obs`)
- **Global coordinate mapping:** `v2/global_map.py` + `v2/map_data.json`
- **Metric logging / what gets summarized:** `v2/tensorboard_callback.py` + `scripts/summarize_sweep_v2.py`
- **Training hyperparameters:** `v2/baseline_fast_v2.py` CLI args and PPO constructor
- **Orchestration:** `scripts/run_sweep_v2.py`, `scripts/run_iterative_sweep_v2.py`

---

## 14) Notes / caveats

- The repo intentionally uses RAM reads for many signals (coords/events/badges/HP). That makes reward shaping and evaluation much more stable than trying to infer everything from pixels.
- Wall-clock termination can interrupt a run between checkpoints. Iterative selection therefore chooses the **latest saved** checkpoint, not an exact “2-hour” boundary.
- Event flag name mapping (`v2/events.json`) is used for logging human-readable flags, but the code notes it may be incomplete/brittle for some flags.
