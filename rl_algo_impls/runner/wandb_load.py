import copy
import os
import shutil
from typing import Tuple, TypeVar

import wandb
from rl_algo_impls.runner.config import Config, Hyperparams, RunArgs

RunArgsType = TypeVar("RunArgsType", bound=RunArgs)


def load_player(
    api: wandb.Api,
    run_path: str,
    args: RunArgsType,
    root_dir: str,
    best: bool,
) -> Tuple[RunArgsType, Config, str]:
    args = copy.copy(args)

    run = api.run(run_path)
    params = run.config
    args.algo = params["algo"]
    args.env = params["env"]
    args.seed = params.get("seed", None)
    args.use_deterministic_algorithms = params.get("use_deterministic_algorithms", True)
    config = Config(args, Hyperparams.from_dict_with_extra_fields(params), root_dir)
    model_path = config.model_dir_path(best=best, downloaded=True)

    model_archive_name = config.model_dir_name(best=best, extension=".zip")
    run.file(model_archive_name).download()
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    shutil.unpack_archive(model_archive_name, model_path)
    os.remove(model_archive_name)

    return args, config, model_path
