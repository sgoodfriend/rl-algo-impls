from dataclasses import asdict
from typing import Any, Dict, Optional

from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.vec_env.procgen import make_procgen_env
from rl_algo_impls.shared.vec_env.vec_env import make_vec_env
from rl_algo_impls.wrappers.self_play_eval_wrapper import SelfPlayEvalWrapper
from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv, find_wrapper


def make_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    if hparams.env_type == "procgen":
        make_env_fn = make_procgen_env
    elif hparams.env_type in {"sb3vec", "gymvec"}:
        make_env_fn = make_vec_env
    elif hparams.env_type == "microrts":
        from rl_algo_impls.microrts.vec_env.microrts import make_microrts_env

        make_env_fn = make_microrts_env
    elif hparams.env_type == "lux":
        from rl_algo_impls.lux.vec_env.lux import make_lux_env

        make_env_fn = make_lux_env
    else:
        raise ValueError(f"env_type {hparams.env_type} not supported")
    return make_env_fn(
        config,
        hparams,
        training=training,
        render=render,
        normalize_load_path=normalize_load_path,
        tb_writer=tb_writer,
    )


def make_eval_env(
    config: Config,
    hparams: EnvHyperparams,
    override_hparams: Optional[Dict[str, Any]] = None,
    self_play_wrapper: Optional[SelfPlayWrapper] = None,
    **kwargs,
) -> VecEnv:
    kwargs = kwargs.copy()
    kwargs["training"] = False
    env_overrides = config.eval_hyperparams.get("env_overrides")
    if env_overrides:
        hparams_kwargs = asdict(hparams)
        hparams_kwargs.update(env_overrides)
        hparams = EnvHyperparams(**hparams_kwargs)
    if override_hparams:
        hparams_kwargs = asdict(hparams)
        for k, v in override_hparams.items():
            hparams_kwargs[k] = v
            if k == "n_envs" and v == 1:
                hparams_kwargs["vec_env_class"] = "sync"
                MAP_PATHS_KEY = "map_paths"
                if hparams_kwargs.get(MAP_PATHS_KEY, []):
                    if len(hparams_kwargs[MAP_PATHS_KEY]) > 1:
                        hparams_kwargs[MAP_PATHS_KEY] = hparams_kwargs[MAP_PATHS_KEY][
                            :1
                        ]
                elif len(hparams_kwargs["make_kwargs"].get(MAP_PATHS_KEY, [])) > 1:
                    hparams_kwargs["make_kwargs"][MAP_PATHS_KEY] = hparams_kwargs[
                        "make_kwargs"
                    ][MAP_PATHS_KEY][:1]
                if hparams_kwargs.get("bots"):
                    one_bot_dict = {}
                    for b, n in hparams_kwargs["bots"].items():
                        one_bot_dict[b] = 1
                        break
                    hparams_kwargs["bots"] = one_bot_dict
        hparams = EnvHyperparams(**hparams_kwargs)
    env = make_env(config, hparams, **kwargs)

    eval_self_play_wrapper = find_wrapper(env, SelfPlayEvalWrapper)
    if eval_self_play_wrapper:
        assert self_play_wrapper
        eval_self_play_wrapper.train_wrapper = self_play_wrapper

    return env
