import optuna

from typing import Any, Dict

from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, single_observation_space


def sample_env_hyperparams(
    trial: optuna.Trial, env_hparams: Dict[str, Any], env: VecEnv
) -> Dict[str, Any]:
    obs_space = single_observation_space(env)

    n_envs = 2 ** trial.suggest_int("n_envs_exp", 1, 5)
    trial.set_user_attr("n_envs", n_envs)
    env_hparams["n_envs"] = n_envs

    normalize = trial.suggest_categorical("normalize", [False, True])
    env_hparams["normalize"] = normalize
    if normalize:
        normalize_kwargs = env_hparams.get("normalize_kwargs", {})
        if len(obs_space.shape) == 3:
            normalize_kwargs.update(
                {
                    "norm_obs": False,
                    "norm_reward": True,
                }
            )
        else:
            norm_obs = trial.suggest_categorical("norm_obs", [True, False])
            norm_reward = trial.suggest_categorical("norm_reward", [True, False])
            normalize_kwargs.update(
                {
                    "norm_obs": norm_obs,
                    "norm_reward": norm_reward,
                }
            )
        env_hparams["normalize_kwargs"] = normalize_kwargs
    elif "normalize_kwargs" in env_hparams:
        del env_hparams["normalize_kwargs"]

    return env_hparams
