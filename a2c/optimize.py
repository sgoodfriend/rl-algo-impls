import optuna

from typing import Any, Dict

from runner.config import Hyperparams


def sample_params(trial: optuna.Trial, base_hyperparams: Hyperparams) -> Hyperparams:
    hyperparams = base_hyperparams.copy()

    # env_hyperparams
    env_hyperparams = hyperparams.get("env_hyperparams", {})

    n_envs = 2 ** trial.suggest_int("n_envs_exp", 1, 5)
    trial.set_user_attr("n_envs", n_envs)
    normalize = trial.suggest_categorical("normalize", [False, True])

    env_hyperparams.update({"n_envs": n_envs, "normalize": normalize})
    hyperparams["env_hyperparams"] = env_hyperparams

    # policy_hyperparams
    policy_hyperparams = hyperparams.get("policy_hyperparams", {})

    # algo_hyperparams
    algo_hyperparams = hyperparams.get("algo_hyperparams", {})

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-3, log=True)
    learning_rate_decay = trial.suggest_categorical(
        "learning_rate_decay", ["none", "linear"]
    )
    n_steps_exp = trial.suggest_int("n_steps_exp", 1, 10)
    n_steps = 2**n_steps_exp
    trial.set_user_attr("n_steps", n_steps)
    gamma = 1.0 - trial.suggest_float("gamma_om", 1e-4, 1e-1, log=True)
    trial.set_user_attr("gamma", gamma)
    gae_lambda = 1 - trial.suggest_float("gae_lambda_om", 1e-4, 1e-1)
    trial.set_user_attr("gae_lambda", gae_lambda)
    ent_coef = trial.suggest_float("ent_coef", 1e-7, 2.5e-2, log=True)
    ent_coef_decay = trial.suggest_categorical("ent_coef_decay", ["none", "linear"])
    vf_coef = trial.suggest_float("vf_coef", 0.2, 0.6)
    max_grad_norm = trial.suggest_float("max_grad_norm", 1e-1, 1e1, log=True)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [True, False])
    normalize_advantage = trial.suggest_categorical(
        "normalize_advantage", [True, False]
    )

    algo_hyperparams.update(
        {
            "learning_rate": learning_rate,
            "learning_rate_decay": learning_rate_decay,
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "ent_coef_decay": ent_coef_decay,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "use_rms_prop": use_rms_prop,
            "normalize_advantage": normalize_advantage,
        }
    )

    if policy_hyperparams.get("use_sde", False):
        sde_sample_freq = 2 ** trial.suggest_int("sde_sample_freq_exp", 0, n_steps_exp)
        trial.set_user_attr("sde_sample_freq", sde_sample_freq)
        algo_hyperparams["sde_sample_freq"] = sde_sample_freq

    hyperparams["algo_hyperparams"] = algo_hyperparams

    return hyperparams
