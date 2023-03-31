import pytest

import gym.spaces
import numpy as np

from rl_algo_impls.shared.policy.actor_critic import clamp_actions


def test_clamp_actions():
    action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    actions = np.array([-1.5, 0, 1.5])
    clamped_actions = clamp_actions(actions, action_space, squash_output=False)
    np.testing.assert_array_equal(clamped_actions, np.array([-1, 0, 1]))

    action_space = gym.spaces.Box(low=-3, high=2, shape=(1,))
    actions = np.array([-1, 0, 1])
    clamped_actions = clamp_actions(actions, action_space, squash_output=True)
    np.testing.assert_array_equal(clamped_actions, np.array([-3, -0.5, 2]))
