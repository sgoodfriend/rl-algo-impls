from typing import Any, Dict

import optuna
from gymnasium.spaces import Box

from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


def sample_on_policy_hyperparams(
    trial: optuna.Trial, policy_hparams: Dict[str, Any], env: VectorEnv
) -> Dict[str, Any]:
    act_space = env.single_action_space

    policy_hparams["init_layers_orthogonal"] = trial.suggest_categorical(
        "init_layers_orthogonal", [True, False]
    )
    policy_hparams["activation_fn"] = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu"]
    )

    if isinstance(act_space, Box):
        policy_hparams["log_std_init"] = trial.suggest_float("log_std_init", -5, 0.5)
        policy_hparams["use_sde"] = trial.suggest_categorical("use_sde", [False, True])

    if policy_hparams.get("use_sde", False):
        policy_hparams["squash_output"] = trial.suggest_categorical(
            "squash_output", [False, True]
        )
    elif "squash_output" in policy_hparams:
        del policy_hparams["squash_output"]

    return policy_hparams
