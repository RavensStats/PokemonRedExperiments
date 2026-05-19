import sys
import argparse
import json
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

def _normalize_checkpoint_path(path_str: str) -> str:
    path_str = (path_str or "").strip()
    if path_str.endswith(".zip"):
        path_str = path_str[:-4]
    return path_str


def make_env(rank, env_conf, seed=0, stream=True, stream_metadata=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        if stream:
            env = StreamWrapper(
                env,
                stream_metadata=(stream_metadata or {
                    "user": "v2-default",
                    "env_id": rank,
                    "color": "#447799",
                    "extra": "",
                }),
            )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def _parse_args():
    parser = argparse.ArgumentParser(description="Pokemon Red RL (V2) trainer")

    # Output / bookkeeping
    parser.add_argument("--run-dir", type=str, default="runs", help="Directory to write checkpoints/TensorBoard logs")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (each env adds its rank)")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from (with or without .zip)")

    # Environment
    parser.add_argument("--num-cpu", type=int, default=4, help="Number of parallel environments (SubprocVecEnv)")
    parser.add_argument("--ep-length", type=int, default=2048 * 80, help="Max steps per episode")
    parser.add_argument("--action-freq", type=int, default=24, help="Emulator ticks per agent action")
    parser.add_argument("--init-state", type=str, default="init.state", help="Initial game state (.state)")
    parser.add_argument("--gb-path", type=str, default="PokemonRed.gb", help="Path to Pokemon Red ROM")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run emulator headless")
    parser.add_argument("--reward-scale", type=float, default=0.5)
    parser.add_argument("--explore-weight", type=float, default=1.0)
    parser.add_argument("--save-trajectory", action=argparse.BooleanOptionalAction, default=True, help="Write training trajectories to session_path")
    parser.add_argument("--trajectory-flush-every", type=int, default=1000, help="Flush trajectory CSV every N env steps (0 disables periodic flushes)")

    # Discovered-events (RAM bit-flip mining)
    parser.add_argument(
        "--discovered-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RAM bit-flip discovery and write discovered_events.json",
    )
    parser.add_argument(
        "--discovered-events-promoted-path",
        type=str,
        default="",
        help="Path to promoted_discovered_events.json to use as frozen shaping list",
    )
    parser.add_argument(
        "--discovered-events-reward-weight",
        type=float,
        default=0.0,
        help="Reward weight for each promoted discovered event when it first occurs",
    )
    parser.add_argument(
        "--discovered-events-flush-every",
        type=int,
        default=500,
        help="Flush discovered_events.json every N env steps (per environment)",
    )

    # Training
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=0,
        help="Total timesteps for model.learn(); default computed from ep-length * num-cpu * 10000",
    )
    parser.add_argument(
        "--train-steps-batch",
        type=int,
        default=0,
        help="PPO n_steps; default ep-length // 64",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device")

    # Streaming (global map broadcast)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="Enable websocket broadcast")
    parser.add_argument("--stream-user", type=str, default="v2-default")
    parser.add_argument("--stream-color", type=str, default="#447799")
    parser.add_argument("--stream-extra", type=str, default="")

    # Logging
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable Weights & Biases logging")

    return parser.parse_args()

if __name__ == "__main__":

    args = _parse_args()

    ep_length = int(args.ep_length)
    sess_path = Path(args.run_dir)
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": bool(args.headless),
        "save_final_state": False,
        "early_stop": False,
        "action_freq": int(args.action_freq),
        "init_state": str(args.init_state),
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(args.gb_path),
        "debug": False,
        "reward_scale": float(args.reward_scale),
        "explore_weight": float(args.explore_weight),
        "save_trajectory": bool(args.save_trajectory),
        "trajectory_flush_every": int(args.trajectory_flush_every),
        "discovered_events_enable": bool(args.discovered_events),
        "discovered_events_promoted_path": str(args.discovered_events_promoted_path),
        "discovered_events_reward_weight": float(args.discovered_events_reward_weight),
        "discovered_events_flush_every": int(args.discovered_events_flush_every),
    }
    
    print(env_config)
    
    num_cpu = int(args.num_cpu)  # Also sets the number of episodes per training iteration
    seed = int(args.seed)
    stream_metadata = {
        "user": str(args.stream_user),
        "color": str(args.stream_color),
        "extra": str(args.stream_extra),
    }
    env = SubprocVecEnv(
        [make_env(i, {**env_config, "instance_id": i}, seed=seed, stream=bool(args.stream), stream_metadata={**stream_metadata, "env_id": i}) for i in range(num_cpu)]
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=sess_path,
        name_prefix="poke",
    )
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if bool(args.wandb):
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=str(sess_path.name),
            name=str(sess_path.name),
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # Checkpoint selection precedence:
    # 1) --resume
    # 2) stdin (non-tty)
    # 3) default: start fresh
    if args.resume:
        file_name = _normalize_checkpoint_path(args.resume)
    elif not sys.stdin.isatty():
        file_name = _normalize_checkpoint_path(sys.stdin.read().strip())
    else:
        file_name = ""

    train_steps_batch = int(args.train_steps_batch) if int(args.train_steps_batch) > 0 else (ep_length // 64)

    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device for training: {device}")
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env, device=device)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            ent_coef=float(args.ent_coef),
            tensorboard_log=str(sess_path),
            device=device,
        )
    
    print(model.policy)

    total_timesteps = int(args.total_timesteps) if int(args.total_timesteps) > 0 else ((ep_length) * num_cpu * 10000)
    class TqdmProgressCallback(BaseCallback):
        def __init__(self, total_timesteps, verbose=0):
            super().__init__(verbose)
            self.total_timesteps = total_timesteps
            self.progress = None
            self.last_step = 0

        def _on_training_start(self) -> None:
            self.progress = tqdm(total=self.total_timesteps, desc="Training Progress", unit="step")
            self.last_step = 0

        def _on_step(self) -> bool:
            current_step = self.model.num_timesteps
            self.progress.update(current_step - self.last_step)
            self.last_step = current_step
            return True

        def _on_training_end(self) -> None:
            self.progress.close()
            print(f"Training completed: {self.total_timesteps} steps.")

    progress_callback = TqdmProgressCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks + [progress_callback]), tb_log_name="poke_ppo")

    if bool(args.wandb):
        run.finish()
