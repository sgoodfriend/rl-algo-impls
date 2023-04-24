import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from rl_algo_impls.runner.config import Config
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.wrappers.action_mask_wrapper import find_action_masker
from rl_algo_impls.wrappers.vectorable_wrapper import (
    VecEnvObs,
    VecEnvStepReturn,
    VectorableWrapper,
)


class SelfPlayWrapper(VectorableWrapper):
    next_obs: VecEnvObs
    next_action_masks: Optional[np.ndarray]

    def __init__(
        self,
        env,
        config: Config,
        num_old_policies: int = 0,
        save_steps: int = 20_000,
        swap_steps: int = 10_000,
        window: int = 10,
        swap_window_size: int = 2,
        selfplay_bots: Optional[Dict[str, Any]] = None,
        bot_always_player_2: bool = False,
    ) -> None:
        super().__init__(env)
        assert num_old_policies % 2 == 0, f"num_old_policies must be even"
        assert (
            num_old_policies % swap_window_size == 0
        ), f"num_old_policies must be a multiple of swap_window_size"

        self.config = config
        self.num_old_policies = num_old_policies
        self.save_steps = save_steps
        self.swap_steps = swap_steps
        self.swap_window_size = swap_window_size
        self.selfplay_bots = selfplay_bots
        self.bot_always_player_2 = bot_always_player_2

        self.policies: Deque[Policy] = deque(maxlen=window)
        self.policy_assignments: List[Optional[Policy]] = [None] * env.num_envs
        self.steps_since_swap = np.zeros(env.num_envs)

        self.selfplay_policies: Dict[str, Policy] = {}

        self.num_envs = env.num_envs - num_old_policies

        if self.selfplay_bots:
            self.num_envs -= sum(self.selfplay_bots.values())
            self.initialize_selfplay_bots()

    def get_action_mask(self) -> Optional[np.ndarray]:
        assert self.next_action_masks is not None
        return self.next_action_masks[self.learner_indexes()]

    def learner_indexes(self) -> List[int]:
        return [p is None for p in self.policy_assignments]

    def checkpoint_policy(self, copied_policy: Policy) -> None:
        copied_policy.train(False)
        self.policies.append(copied_policy)

        if all(p is None for p in self.policy_assignments[: 2 * self.num_old_policies]):
            for i in range(self.num_old_policies):
                # Switch between player 1 and 2
                self.policy_assignments[
                    2 * i + (i % 2 if not self.bot_always_player_2 else 1)
                ] = copied_policy

    def swap_policy(self, idx: int, swap_window_size: int = 1) -> None:
        policy = random.choice(self.policies)
        idx = idx // 2 * 2
        for j in range(swap_window_size * 2):
            if self.policy_assignments[idx + j]:
                self.policy_assignments[idx + j] = policy
        self.steps_since_swap[idx : idx + swap_window_size * 2] = np.zeros(
            swap_window_size * 2
        )

    def initialize_selfplay_bots(self) -> None:
        if not self.selfplay_bots:
            return
        from rl_algo_impls.runner.running_utils import get_device, make_policy

        env = self.env  # Type: ignore
        device = get_device(self.config, env)
        start_idx = 2 * self.num_old_policies
        for model_path, n in self.selfplay_bots.items():
            policy = make_policy(
                self.config,
                env,
                device,
                load_path=model_path,
                **self.config.policy_hyperparams,
            ).eval()
            self.selfplay_policies["model_path"] = policy
            for idx in range(start_idx, start_idx + 2 * n, 2):
                bot_idx = (
                    (idx + 1) if self.bot_always_player_2 else (idx + idx // 2 % 2)
                )
                self.policy_assignments[bot_idx] = policy
            start_idx += 2 * n

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        env = self.env  # type: ignore
        all_actions = np.zeros((env.num_envs,) + actions.shape[1:], dtype=actions.dtype)
        orig_learner_indexes = self.learner_indexes()

        all_actions[orig_learner_indexes] = actions
        for policy in set(p for p in self.policy_assignments if p):
            policy_indexes = [policy == p for p in self.policy_assignments]
            if any(policy_indexes):
                all_actions[policy_indexes] = policy.act(
                    self.next_obs[policy_indexes],
                    deterministic=False,
                    action_masks=self.next_action_masks[policy_indexes]
                    if self.next_action_masks is not None
                    else None,
                )
        self.next_obs, rew, done, info = env.step(all_actions)
        self.next_action_masks = env.get_action_mask()

        rew = rew[orig_learner_indexes]
        info = [i for i, b in zip(info, orig_learner_indexes) if b]

        self.steps_since_swap += 1
        for idx in range(0, self.num_old_policies * 2, 2 * self.swap_window_size):
            if self.steps_since_swap[idx] > self.swap_steps:
                self.swap_policy(idx, self.swap_window_size)

        new_learner_indexes = self.learner_indexes()
        return self.next_obs[new_learner_indexes], rew, done[new_learner_indexes], info

    def reset(self) -> VecEnvObs:
        self.next_obs = super().reset()
        self.next_action_masks = self.env.get_action_mask()
        return self.next_obs[self.learner_indexes()]
