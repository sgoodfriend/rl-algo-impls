from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class EnvHyperparams:
    env_type: str = "gymvec"
    n_envs: int = 1
    frame_stack: int = 1
    make_kwargs: Optional[Dict[str, Any]] = None
    no_reward_timeout_steps: Optional[int] = None
    no_reward_fire_steps: Optional[int] = None
    vec_env_class: str = "sync"
    normalize: bool = False
    normalize_kwargs: Optional[Dict[str, Any]] = None
    rolling_length: int = 100
    video_step_interval: Union[int, float] = 1_000_000
    initial_steps_to_truncate: Optional[int] = None
    clip_atari_rewards: bool = True
    normalize_type: Optional[str] = None
    mask_actions: bool = False
    bots: Optional[Dict[str, int]] = None
    self_play_kwargs: Optional[Dict[str, Any]] = None
    selfplay_bots: Optional[Dict[str, int]] = None
    additional_win_loss_reward: bool = False
    map_paths: Optional[List[str]] = None
    score_reward_kwargs: Optional[Dict[str, int]] = None
    is_agent: bool = False
    valid_sizes: Optional[List[int]] = None
    paper_planes_sizes: Optional[List[int]] = None
    fixed_size: bool = False
    terrain_overrides: Optional[Dict[str, Any]] = None
    time_budget_ms: Optional[int] = None
    video_frames_per_second: Optional[int] = None
    reference_bot: Optional[str] = None
    play_checkpoints_kwargs: Optional[Dict[str, Any]] = None
    additional_win_loss_smoothing_factor: Optional[float] = None
    info_rewards: Optional[Dict[str, Any]] = None