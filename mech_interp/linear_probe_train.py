import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from torch.utils.tensorboard.writer import SummaryWriter

from mech_interp.occupancy_linear_probe_trainer import OccupancyLinearProbeTrainer
from rl_algo_impls.runner.config import Config, Hyperparams, RunArgs
from rl_algo_impls.runner.env_hyperparams import EnvHyperparams
from rl_algo_impls.runner.running_utils import (
    base_parser,
    load_hyperparams,
    make_eval_policy,
    set_device_optimizations,
    set_seeds,
)
from rl_algo_impls.shared.data_store.data_store_view import EvalDataStoreView
from rl_algo_impls.shared.data_store.in_process_data_store_accessor import (
    InProcessDataStoreAccessor,
)
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.policy.actor_critic_network.grid2entity_transformer import (
    Grid2EntityTransformerNetwork,
)
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_eval_env
from rl_algo_impls.utils.device import get_device


@dataclass
class LinearProbeTrainArgs(RunArgs):
    best: bool = True
    wandb_run_path: Optional[str] = None
    override_hparams: Optional[Dict[str, Any]] = None
    n_envs: Optional[int] = 1
    deterministic_eval: Optional[bool] = None
    n_steps: int = 10_000
    no_print_returns: bool = False

    run_prefix: str = "probe"
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_group: Optional[str] = None

    detach_from_model: bool = False

    learning_rate: float = 3e-4


def linear_probe_train() -> None:
    parser = base_parser(multiple=False)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument("--n_steps", default=20_000, type=int)
    parser.add_argument(
        "--no-print-returns", action="store_true", help="Limit printing"
    )
    # wandb-run-path overrides base RunArgs
    parser.add_argument("--wandb-run-path", default=None, type=str)
    parser.add_argument("--override-hparams", default=None, type=str)
    parser.add_argument("--run-prefix", default="probe", type=str)
    parser.add_argument(
        "--wandb-project-name", default="rl-algo-impls-interp", type=str
    )
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-tags", default=None, nargs="*")
    parser.add_argument("--wandb-group", default=None, type=str)
    parser.add_argument("--detach-from-model", action="store_true")
    parser.add_argument("--learning-rate", default=3e-4, type=float)

    parser.set_defaults(
        n_envs=8,
        wandb_run_path="sgoodfriend/mech-interp-rl-algo-impls/eh5nxxe2",
        override_hparams='{"bots":{"coacAI":8}, "map_paths": ["maps/10x10/basesTwoWorkers10x10.xml"]}',
        detach_from_model=False,
        learning_rate=5e-3,
    )
    args = parser.parse_args()
    args.algo = args.algo[0]
    args.env = args.env[0]
    args.seed = args.seed[0]
    args.override_hparams = (
        json.loads(args.override_hparams) if args.override_hparams else None
    )
    args = LinearProbeTrainArgs(**vars(args))

    train(args, os.getcwd())


def train(args: LinearProbeTrainArgs, root_dir: str) -> None:
    if args.wandb_run_path:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        params = run.config

        args.algo = params["algo"]
        args.env = params["env"]
        args.seed = params.get("seed", None)

        config = Config(args, Hyperparams.from_dict_with_extra_fields(params), root_dir)
        model_path = config.model_dir_path(best=args.best, downloaded=True)

        model_archive_name = config.model_dir_name(best=args.best, extension=".zip")
        run.file(model_archive_name).download()
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.unpack_archive(model_archive_name, model_path)
        os.remove(model_archive_name)
    else:
        hyperparams = load_hyperparams(args.algo, args.env)

        config = Config(args, hyperparams, root_dir)
        model_path = config.model_dir_path(best=args.best)

    print(args)

    run_prefix = args.run_prefix
    if args.detach_from_model:
        run_prefix += "-detached"
    run_name = f"{run_prefix}-{config.model_name()}-{datetime.now().isoformat()}"
    tb_path = os.path.join(root_dir, "runs", run_name)
    wandb_enabled = bool(args.wandb_project_name)
    if wandb_enabled:
        wandb.tensorboard.patch(root_logdir=tb_path, pytorch=True)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=asdict(args),
            name=run_name,
            tags=args.wandb_tags,
            group=args.wandb_group,
        )
    tb_writer = SummaryWriter(tb_path)

    set_seeds(args.seed)

    data_store_accessor = InProcessDataStoreAccessor(
        **(config.hyperparams.checkpoints_kwargs or {})
    )

    override_hparams = args.override_hparams or {}
    if args.n_envs:
        override_hparams["n_envs"] = args.n_envs
    env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        EvalDataStoreView(data_store_accessor, is_eval_job=True),
        override_hparams=override_hparams,
    )

    device = get_device(config, EnvSpaces.from_vec_env(env))
    set_device_optimizations(device, **config.device_hyperparams)
    policy = make_eval_policy(
        config,
        EnvSpaces.from_vec_env(env),
        device,
        data_store_accessor,
        load_path=model_path,
        **config.policy_hyperparams,
    ).eval()

    deterministic = (
        args.deterministic_eval
        if args.deterministic_eval is not None
        else config.eval_hyperparams.get("deterministic", True)
    )

    assert isinstance(policy, ActorCritic)
    network = policy.network
    assert isinstance(network, Grid2EntityTransformerNetwork)
    occupancy_trainer = OccupancyLinearProbeTrainer(
        device,
        network.encoder_embed_dim,
        env.observation_space.shape[-2:],
        learning_rate=args.learning_rate,
        detach=args.detach_from_model,
    )
    occupancy_trainer.train(
        policy,
        env,
        args.n_steps,
        tb_writer,
        deterministic_actions=deterministic,
    )

    tb_writer.close()
