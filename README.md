# Train RL agents to play Pokemon Red

### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section
  
## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

#### Running Many Experiments Overnight (Sweep)

This repo already supports long-running training with checkpoints + TensorBoard logs. To run *many* independent runs (e.g., different seeds) overnight, use:

```bash
python scripts/run_sweep_v2.py --runs 10 --seed-start 0 --num-cpu 4 --device auto --no-stream
```

To run for a wall-clock budget (stop launching new runs after ~N hours):

```bash
python scripts/run_sweep_v2.py --runs 9999 --seed-start 0 --hours 8 --num-cpu 4 --device auto --no-stream
```

#### Iterative Training (2h -> pick best -> resume -> repeat)

If you want an automatic "tournament" loop (run a short stage, pick the best run by summary metrics, resume from its latest checkpoint, repeat), use:

```bash
python scripts/run_iterative_sweep_v2.py --stages 6 --stage-hours 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --device auto --no-stream
```

Modes (dynamic discovered-event reward shaping):

- Explicit baseline (default):

```bash
python scripts/run_iterative_sweep_v2.py --mode explicit --stages 6 --stage-hours-0 4 --stage-hours-rest 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --device auto --no-stream
```

- Discovered events, ranked by explicit metrics (badge -> events_completed -> explore -> max_level):

```bash
python scripts/run_iterative_sweep_v2.py --mode discovered_explicit --stages 6 --stage-hours-0 4 --stage-hours-rest 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --device auto --no-stream
```

- Discovered events, ranked ignoring explicit `events_completed` (badge -> explore -> max_level):

```bash
python scripts/run_iterative_sweep_v2.py --mode discovered_no_explicit --stages 6 --stage-hours-0 4 --stage-hours-rest 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --device auto --no-stream
```

In `discovered_*` modes, each stage writes a `promoted_discovered_events.json` (selected from top runs) and the next stage uses it as a frozen shaping list. Each run also writes `discovered_events_env<id>.json` snapshots in its run folder.

To run all three modes as separate processes (optionally in parallel):

```bash
python scripts/run_iterative_compare_v2.py --max-parallel-modes 3 --base-tag compare --stages 6 --stage-hours-0 4 --stage-hours-rest 2 --initial-runs 10 --initial-max-parallel 1 --num-cpu 4 --device auto --no-stream
```

Pause/resume:

- For `run_iterative_sweep_v2.py`, pass a fixed `--root-dir` and use Ctrl+C to stop; later re-run with `--resume-root`.
- For `run_iterative_compare_v2.py`, add `--resume` so each mode uses a stable root folder and will resume if interrupted.

Example (resume-friendly, sequential modes):

```bash
python scripts/run_iterative_compare_v2.py --max-parallel-modes 1 --base-tag fair_fast --resume --root-base-dir sweeps --stages 6 --stage-hours-0 4 --stage-hours-rest 2 --initial-runs 10 --initial-max-parallel 2 --seeds 0,1,2,3,4,5,6,7,8,9 --continue-runs 1 --num-cpu 4 --device cuda --no-stream
```

Each stage writes `summary.csv`, `best_run_dir.txt`, and `best_checkpoint.txt` under the stage folder.

It creates a timestamped folder under `sweeps/`, with one subfolder per run (each contains checkpoints, TensorBoard logs, and `stdout.log`). By default, full output is stored losslessly in `stdout.log.gz`, and `stdout.log` is a small pointer file.

To generate a single CSV summary (Badge Count, Map Explored %, Events Completed, Highest Pokemon Level) after the sweep finishes:

```bash
python scripts/summarize_sweep_v2.py sweeps/<your_sweep_folder>
```

To generate efficiency plots (metric vs timesteps) and an efficiency CSV (AUC + optional time-to-threshold):

```bash
python scripts/plot_sweep_v2.py sweeps/<your_sweep_folder> --top-k 10 --rank-by badge_count --threshold badge_count=1
```

To add explicit "time-to-goal" reporting (e.g., time to badge 1/2/3, time to 10/25 events):

```bash
python scripts/plot_sweep_v2.py sweeps/<your_sweep_folder> --goals badge_count=1|2|3,events_completed=10|25,map_explored_pct=1|2,highest_pokemon_level=10|15
```

Relative-to-final goals (25%/50%/75% of each run's own final value) are enabled by default via `--relative-goals 0.25|0.5|0.75`.

This writes two CSVs in `plots/`:
- `efficiency_auc.csv` (AUC + final values)
- `efficiency_badge_times.csv` (timesteps to badge 1/2/3)

It also writes:
- `efficiency_goal_times.csv` (timesteps to each absolute/relative goal for each metric)

`efficiency_goal_times.csv` is written pre-sorted (fastest-to-goals first). Goals not reached are filled with a large sentinel timestep value so Excel sorting works.

It also writes goal-time graphs in `plots/`:
- `goal_times_heatmap.png` (runs x goals)
- `goal_timesteps_to_...png` (one per goal column)

Note: Map Explored % is computed over the union of stitched map rectangles from `v2/map_data.json` (so it ignores padding/outside-of-map tiles).

You can also run a single V2 training run with explicit output folder:

```bash
python v2/baseline_fast_v2.py --run-dir sweeps/my_run --seed 123 --device auto --no-stream
```

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Follow up work  
 
Check out our follow up projects & papers!  
  
### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
