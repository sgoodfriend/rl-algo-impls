import inspect
import logging
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, Optional

from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from rl_algo_impls.runner.config import Config, TrainArgs
from rl_algo_impls.shared.summary_wrapper.abstract_summary_wrapper import (
    AbstractSummaryWrapper,
)


class InProcessSummaryWrapper(AbstractSummaryWrapper):
    def __init__(self, config: Config, args: TrainArgs) -> None:
        super().__init__()
        self.config = config
        self.wandb_enabled = bool(args.wandb_project_name)
        if self.wandb_enabled:
            wandb.tensorboard.patch(
                root_logdir=config.tensorboard_summary_path, pytorch=True
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=asdict(config.hyperparams),
                name=config.run_name(),
                # monitor_gym=True,
                # save_code=True,
                tags=args.wandb_tags,
                group=args.wandb_group,
            )
            wandb.config.update(args)

        self.tb_writer = SummaryWriter(config.tensorboard_summary_path)
        self.timesteps_elapsed = 0

    def on_timesteps_elapsed(self, timesteps_elapsed: int) -> None:
        self.timesteps_elapsed = timesteps_elapsed

    def close(self) -> None:
        self.tb_writer.close()

        if self.wandb_enabled:
            self.make_wandb_archive(self.config.model_dir_path())
            wandb.finish()

    def __getattr__(self, name: str):
        attr = getattr(self.tb_writer, name)
        if callable(attr) and "global_step" in inspect.signature(attr).parameters:

            def wrapper(*args, **kwargs):
                attr(*args, global_step=self.timesteps_elapsed, **kwargs)

            return wrapper
        else:
            return attr

    def update_summary(self, summary_update: Dict[str, Any]) -> None:
        for k, v in summary_update.items():
            if self.wandb_enabled:
                wandb.run.summary[k] = v  # type: ignore
            else:
                logging.info(f"{k} = {v}")

    def make_wandb_archive(self, path: str) -> None:
        if self.wandb_enabled:
            filename = os.path.split(path)[-1]
            shutil.make_archive(
                os.path.join(wandb.run.dir, filename),  # type: ignore
                "zip",
                path,
            )

    def log_video(self, video_path: str, fps: int) -> None:
        if self.wandb_enabled:
            wandb.log(
                {
                    "videos": wandb.Video(video_path, fps=fps),
                },
                step=self.timesteps_elapsed,
            )

    def maybe_add_logging_handler(self, logger: Optional[logging.Logger]) -> None:
        pass
