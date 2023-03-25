from dataclasses import asdict
from typing import Optional

from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.vec_env.microrts import make_microrts_env
from rl_algo_impls.shared.vec_env.procgen import make_procgen_env
from rl_algo_impls.shared.vec_env.vec_env import make_vec_env
from rl_algo_impls.wrappers.vectorable_wrapper import VecEnv


def make_env(
    config: Config,
    hparams: EnvHyperparams,
    training: bool = True,
    render: bool = False,
    normalize_load_path: Optional[str] = None,
    tb_writer: Optional[SummaryWriter] = None,
) -> VecEnv:
    if hparams.env_type == "procgen":
        return make_procgen_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    elif hparams.env_type in {"sb3vec", "gymvec"}:
        return make_vec_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    elif hparams.env_type == "microrts":
        return make_microrts_env(
            config,
            hparams,
            training=training,
            render=render,
            normalize_load_path=normalize_load_path,
            tb_writer=tb_writer,
        )
    else:
        raise ValueError(f"env_type {hparams.env_type} not supported")


def make_eval_env(
    config: Config,
    hparams: EnvHyperparams,
    override_n_envs: Optional[int] = None,
    **kwargs,
) -> VecEnv:
    kwargs = kwargs.copy()
    kwargs["training"] = False
    if override_n_envs is not None:
        hparams_kwargs = asdict(hparams)
        hparams_kwargs["n_envs"] = override_n_envs
        if override_n_envs == 1:
            hparams_kwargs["vec_env_class"] = "sync"
        hparams = EnvHyperparams(**hparams_kwargs)
    return make_env(config, hparams, **kwargs)