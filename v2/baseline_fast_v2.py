import sys
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

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf), 
            stream_metadata = { # All of this is part is optional
                "user": "v2-default", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#447799", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': 'init.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
            }
    
    print(env_config)
    
    num_cpu = 4 # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from    
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip() #"runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64
    
    import torch
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
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
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path, device=device)
    
    print(model.policy)

    total_timesteps = (ep_length) * num_cpu * 10000
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

    if use_wandb_logging:
        run.finish()
