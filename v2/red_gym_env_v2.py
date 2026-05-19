import uuid
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE, VALID_TILE_MASK

event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)

# Ordered game milestones: (address, mask, point_value)
# Point values increase toward the end to incentivize full game completion.
GAME_MILESTONES = [
    (0xD74B, 0x04,  1),   # Starter Received
    (0xD74B, 0x20,  2),   # Pokedex Received
    (0xD755, 0x80,  3),   # Beat Brock (Gym 1)
    (0xD7EB, 0x01,  4),   # Got SS Ticket
    (0xD75E, 0x80,  5),   # Beat Misty (Gym 2)
    (0xD772, 0x02,  6),   # Got HM01 Cut
    (0xD773, 0x80,  7),   # Beat Lt. Surge (Gym 3)
    (0xD7C2, 0x01,  8),   # Got HM05 Flash
    (0xD77C, 0x02,  9),   # Beat Erika (Gym 4)
    (0xD77E, 0x02, 10),   # Found Rocket Hideout
    (0xD825, 0x20, 11),   # Got Silph Scope
    (0xD768, 0x80, 12),   # Beat Marowak Ghost
    (0xD76C, 0x01, 13),   # Got Poke Flute
    (0xD792, 0x02, 14),   # Beat Koga (Gym 5)
    (0xD790, 0x80, 15),   # Got HM03 Surf
    (0xD838, 0x80, 16),   # Silph Co Liberated
    (0xD7B3, 0x02, 17),   # Beat Sabrina (Gym 6)
    (0xD796, 0x01, 18),   # Mansion Switch On
    (0xD79A, 0x02, 19),   # Beat Blaine (Gym 7)
    (0xD751, 0x02, 20),   # Beat Giovanni (Gym 8)
    (0xD796, 0x1E, 21),   # Beat Elite Four (all 4 members)
    (0xD796, 0x20, 22),   # Beat Champion
]

class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.save_trajectory = bool(config.get("save_trajectory", False))
        self.trajectory_flush_every = int(config.get("trajectory_flush_every", 0))
        self.frame_stacks = 3
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.trajectory_path = None
        if self.save_trajectory:
            try:
                instance_num = int(self.instance_id)
                traj_name = "trajectory.csv.gz" if instance_num == 0 else f"trajectory_worker_{instance_num}.csv.gz"
            except Exception:
                traj_name = f"trajectory_{self.instance_id}.csv.gz"
            self.trajectory_path = self.s_path / traj_name
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("v2/events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.enc_freqs = 8

        # Input jitter (action noise) for robustness experiments
        self.input_jitter_enable = bool(config.get("input_jitter_enable", False))
        self.input_jitter_prob = float(config.get("input_jitter_prob", 0.0))
        self.input_jitter_mode = str(config.get("input_jitter_mode", "lag")).strip().lower()

        # Perception noise (sensor error) applied to the *observed* position/map crop
        # This perturbs only the explore-map observation, not the underlying game state or reward.
        self.perception_noise_enable = bool(config.get("perception_noise_enable", False))
        self.perception_noise_radius = int(config.get("perception_noise_radius", 0))
        self.perception_noise_mode = str(config.get("perception_noise_mode", "uniform")).strip().lower()

        # Discovered-events (RAM bit-flip mining)
        self.discovered_events_enable = bool(config.get("discovered_events_enable", False))
        # List of (start, end) address ranges (inclusive start, exclusive end), or default to event-flag region.
        self.discovered_events_ranges = config.get(
            "discovered_events_ranges",
            [(event_flags_start, event_flags_end)],
        )
        self.discovered_events_min_address = int(config.get("discovered_events_min_address", 0))
        self.discovered_events_max_address = int(config.get("discovered_events_max_address", 0x10000))
        self.discovered_events_flush_every = int(config.get("discovered_events_flush_every", 500))

        # Promoted events file (frozen shaping list for this run)
        promoted_path = str(config.get("discovered_events_promoted_path", "")).strip()
        self.discovered_events_promoted_path = promoted_path
        self.discovered_events_reward_weight = float(config.get("discovered_events_reward_weight", 0.0))
        self.promoted_discovered_events: Dict[str, float] = {}

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad*4,self.coords_pad*4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        head = "null" if config["headless"] else "SDL2"

        #log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            #debugging=False,
            #disable_input=False,
            window=head,
        )

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

        # Load promoted discovered events (frozen shaping list)
        if self.discovered_events_promoted_path:
            try:
                promoted = json.loads(Path(self.discovered_events_promoted_path).read_text(encoding="utf-8"))
                events = promoted.get("events", []) if isinstance(promoted, dict) else []
                for item in events:
                    if not isinstance(item, dict):
                        continue
                    eid = str(item.get("id", "")).strip()
                    w = float(item.get("weight", 1.0))
                    if eid:
                        self.promoted_discovered_events[eid] = w
            except Exception:
                # If the file isn't readable, just run without shaping.
                self.promoted_discovered_events = {}

    def reset(self, seed=None, options={}):
        self.seed = seed
        # RNG for deterministic jitter across runs (seed provided by SubprocVecEnv)
        try:
            self._rng = np.random.default_rng(seed)
        except Exception:
            self._rng = np.random.default_rng(None)

        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)
        self.valid_tile_mask = VALID_TILE_MASK

        self.recent_screens = np.zeros( self.output_shape, dtype=np.uint8)
        
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        # Jitter bookkeeping
        self._prev_action = 0
        self.input_jitter_count = 0
        self.input_jitter_last_applied = False

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.opponent_damage_reward = 0
        self.last_opponent_hp = 0
        self.prev_is_in_battle = 0
        self.level_penalty_total = 0
        self.max_game_completion_score = 0

        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        # Discovered-events state
        self.discovered_events: Dict[str, Dict] = {}
        self._discovered_prev_bytes: Optional[Dict[int, int]] = None
        self.discovered_event_reward_total = 0.0
        self.discovered_event_reward_max = 0.0
        self.discovered_events_last_flush_step = 0

        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def flush_trajectory(self):
        if not self.save_trajectory or self.trajectory_path is None or not self.agent_stats:
            return
        try:
            df = pd.DataFrame(self.agent_stats)
            tmp_path = self.trajectory_path.with_name(self.trajectory_path.name + ".tmp")
            df.to_csv(tmp_path, index=False, compression="gzip")
            tmp_path.replace(self.trajectory_path)
        except Exception:
            pass

    def _apply_input_jitter(self, action: int) -> Tuple[int, bool]:
        """Optionally perturb the chosen action to simulate input lag/drift.

        Returns (applied_action, jitter_applied).
        """
        if not self.input_jitter_enable:
            return action, False
        p = float(self.input_jitter_prob)
        if p <= 0:
            return action, False

        # Decide whether to apply jitter this step
        if float(self._rng.random()) >= p:
            return action, False

        mode = self.input_jitter_mode

        # Default: lag/sticky (repeat previous action)
        if mode in ("lag", "sticky", "repeat"):
            return int(self._prev_action), True

        # Drift: if moving, randomly change direction
        if mode in ("drift", "direction"):
            # Arrow actions are indices 0..3 in valid_actions
            if 0 <= int(action) <= 3:
                choices = [0, 1, 2, 3]
                try:
                    choices.remove(int(action))
                except Exception:
                    pass
                return int(self._rng.choice(choices)), True
            return action, False

        # Random: replace with a random valid action
        if mode in ("random", "rand"):
            return int(self._rng.integers(0, len(self.valid_actions))), True

        # Unknown mode -> fall back to lag
        return int(self._prev_action), True

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):
        
        screen = self.render()

        self.update_recent_screens(screen)
        
        # normalize to approx 0-1
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions
        }

        return observation

    def step(self, action):

        action = int(action)

        if self.save_video and self.step_count == 0:
            self.start_video()

        requested_action = action
        applied_action, jitter_applied = self._apply_input_jitter(action)
        self.input_jitter_last_applied = bool(jitter_applied)
        if jitter_applied:
            self.input_jitter_count += 1

        self.run_action_on_emulator(applied_action)
        self.append_agent_stats(applied_action)

        # Track both requested and applied actions
        if self.agent_stats:
            self.agent_stats[-1]["requested_action"] = int(requested_action)
            self.agent_stats[-1]["applied_action"] = int(applied_action)
            self.agent_stats[-1]["input_jitter_applied"] = int(bool(jitter_applied))
            self.agent_stats[-1]["input_jitter_count"] = int(self.input_jitter_count)

        self.update_recent_actions(applied_action)
        self._prev_action = int(applied_action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.update_battle_rewards()

        self.party_size = self.read_m(0xD163)

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        # self.save_and_print_info(step_limit_reached, obs)

        # create a map of all event flags set, with names where possible
        #if step_limit_reached:
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        # TODO this currently seems to be broken!
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1

        if self.save_trajectory and self.trajectory_path is not None:
            if step_limit_reached or (self.trajectory_flush_every > 0 and self.step_count % self.trajectory_flush_every == 0):
                self.flush_trajectory()

        # Discovered-events: update after step_count increments and flush periodically
        if self.discovered_events_enable:
            self.update_discovered_events()
            if self.discovered_events_flush_every > 0 and (self.step_count - self.discovered_events_last_flush_step) >= self.discovered_events_flush_every:
                self.flush_discovered_events()
                self.discovered_events_last_flush_step = self.step_count
            if step_limit_reached:
                self.flush_discovered_events()

        return obs, new_reward, False, step_limit_reached, {}

    def _iter_discovery_addresses(self):
        for start, end in self.discovered_events_ranges:
            s = max(int(start), self.discovered_events_min_address)
            e = min(int(end), self.discovered_events_max_address)
            for addr in range(s, e):
                yield addr

    def update_discovered_events(self) -> None:
        """Mine 'events' as RAM bit flips (0->1) across configured address ranges.

        We store them as metrics and (optionally) use *promoted* discovered events as shaping.
        """
        # Build prev snapshot on first call
        if self._discovered_prev_bytes is None:
            self._discovered_prev_bytes = {addr: int(self.read_m(addr)) for addr in self._iter_discovery_addresses()}
            return

        prev = self._discovered_prev_bytes
        for addr in list(prev.keys()):
            cur = int(self.read_m(addr))
            old = int(prev[addr])
            if cur == old:
                continue
            # Detect 0->1 bit flips
            flipped_on = (~old) & cur
            if flipped_on:
                for bit in range(8):
                    if flipped_on & (1 << bit):
                        eid = f"bit:0x{addr:04X}:{bit}"
                        if eid not in self.discovered_events:
                            self.discovered_events[eid] = {
                                "id": eid,
                                "addr": int(addr),
                                "bit": int(bit),
                                "first_step": int(self.step_count),
                            }
                            # Optional shaping (only for promoted list, to keep reward stable per stage)
                            if self.discovered_events_reward_weight > 0 and eid in self.promoted_discovered_events:
                                self.discovered_event_reward_total += (
                                    float(self.discovered_events_reward_weight) * float(self.promoted_discovered_events[eid])
                                )

            prev[addr] = cur

        # Track a non-decreasing shaping component (max over time)
        self.discovered_event_reward_max = max(self.discovered_event_reward_max, float(self.discovered_event_reward_total))

    def flush_discovered_events(self) -> None:
        """Write a lossless snapshot to <run_dir>/discovered_events_env<id>.json.

        This is overwritten each flush so orchestration scripts can read partial progress.
        """
        try:
            out_path = self.s_path / Path(f"discovered_events_env{self.instance_id}.json")
            payload = {
                "step": int(self.step_count),
                "count": int(len(self.discovered_events)),
                "events": sorted(self.discovered_events.values(), key=lambda e: int(e.get("first_step", 0))),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()
        
    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        explore_tiles = int(((self.explore_map > 0) & self.valid_tile_mask).sum())
        total_tiles = int(self.valid_tile_mask.sum())
        explore_pct = 100.0 * explore_tiles / float(max(total_tiles, 1))
        events_completed = int(self.get_all_events_reward())
        game_completion_score = int(self.get_game_completion_score())
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "max_level": int(max(levels) if levels else 0),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "explore_tiles": explore_tiles,
                "explore_total_tiles": total_tiles,
                "explore_pct": explore_pct,
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "events_completed": events_completed,
                "game_completion_score": game_completion_score,
                "game_completion_max": int(self.max_game_completion_score),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "discovered_event_count": int(len(self.discovered_events)) if self.discovered_events_enable else 0,
                "discovered_event_reward": float(self.discovered_event_reward_max) if self.discovered_events_enable else 0.0,
                "input_jitter_enable": int(bool(self.input_jitter_enable)),
                "input_jitter_prob": float(self.input_jitter_prob),
                "input_jitter_mode": str(self.input_jitter_mode),
                "input_jitter_count": int(getattr(self, "input_jitter_count", 0)),
            }
        )

    def start_video(self):

        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad*4, self.coords_pad*4), 
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        # if not in battle
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1
            #self.seen_coords[coord_string] = self.step_count

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if self.perception_noise_enable and int(self.perception_noise_radius) > 0:
            r = int(self.perception_noise_radius)
            mode = self.perception_noise_mode
            if mode == "normal":
                dy = int(self._rng.normal(0.0, float(r)))
                dx = int(self._rng.normal(0.0, float(r)))
            else:
                # uniform integer noise in [-r, r]
                dy = int(self._rng.integers(-r, r + 1))
                dx = int(self._rng.integers(-r, r + 1))
            c = (int(c[0]) + dy, int(c[1]) + dx)
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0]+self.coords_pad,
                c[1]-self.coords_pad:c[1]+self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:,:, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:,:, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    def read_m(self, addr):
        #return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def get_game_completion_score(self):
        """Sum reward points for each achieved game milestone."""
        return sum(
            pts
            for addr, mask, pts in GAME_MILESTONES
            if (self.read_m(addr) & mask) == mask
        )

    def update_game_completion_rew(self):
        """Track max game completion score (never decreases)."""
        cur_score = self.get_game_completion_score()
        self.max_game_completion_score = max(self.max_game_completion_score, cur_score)
        return self.max_game_completion_score

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            #"level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 10,
            #"op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            #"dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 100,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.15,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05,
            "op_damage": self.reward_scale * self.opponent_damage_reward * 2,
            "level_pen": self.reward_scale * self.level_penalty_total * -0.5,
            "game_progress": self.reward_scale * self.update_game_completion_rew() * 5,
            "discovered": self.reward_scale * float(self.discovered_event_reward_max),
        }

        return state_scores

    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_opponent_hp_fraction(self):
        hp = self.read_hp(0xCFE6)       # wEnemyMonHP (2 bytes)
        max_hp = self.read_hp(0xCFF4)   # wEnemyMonMaxHP (2 bytes)
        max_hp = max(max_hp, 1)
        return hp / max_hp

    def update_battle_rewards(self):
        """Track opponent HP damage reward and level penalty at battle start."""
        is_in_battle = self.read_m(0xD057)

        # Level penalty: fired once when a new battle begins
        if is_in_battle and not self.prev_is_in_battle:
            player_level = self.read_m(0xD18C)   # Party slot 0 level
            enemy_level = self.read_m(0xCFF3)    # wEnemyMonLevel
            if enemy_level > 0 and player_level <= enemy_level:
                self.level_penalty_total += 1

        # Opponent damage reward: accumulate HP loss dealt to current enemy
        if is_in_battle:
            cur_opp_hp = self.read_opponent_hp_fraction()
            # Detect a new opponent (HP jumped up significantly)
            if cur_opp_hp > self.last_opponent_hp + 0.3:
                self.last_opponent_hp = cur_opp_hp
            hp_decrease = self.last_opponent_hp - cur_opp_hp
            if hp_decrease > 0:
                self.opponent_damage_reward += hp_decrease
                self.last_opponent_hp = cur_opp_hp
        else:
            self.last_opponent_hp = 0

        self.prev_is_in_battle = is_in_battle

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))
    
    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
