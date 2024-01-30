import inspect
import logging
import os
import shutil
from dataclasses import asdict
from typing import Any, Dict, Set

import ray
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from rl_algo_impls.runner.config import Config, TrainArgs


@ray.remote
class SummaryWriterActor:
    def __init__(self, config: Config, args: TrainArgs):
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
        self.paths_live_saving: Set[str] = set()

    def close(self) -> None:
        self.tb_writer.close()
        if self.wandb_enabled:
            self.make_wandb_archive(self.config.model_dir_path())
            wandb.finish()

    def call(self, name: str, *args, **kwargs):
        attr = getattr(self.tb_writer, name)
        if "global_step" not in inspect.signature(attr).parameters:
            kwargs.pop("global_step", None)
        return attr(*args, **kwargs)

    def update_summary(self, summary_update: Dict[str, Any]) -> None:
        for k, v in summary_update.items():
            if self.wandb_enabled:
                wandb.run.summary[k] = v  # type: ignore
            else:
                logging.info(f"{k} = {v}")

    def make_wandb_archive(self, path: str) -> None:
        if self.wandb_enabled:
            filename = os.path.split(path)[-1]
            archive_path = os.path.join(wandb.run.dir, filename)  # type: ignore
            shutil.make_archive(
                archive_path,
                "zip",
                path,
            )
            save_path = f"{archive_path}.zip"
            if not save_path in self.paths_live_saving:
                self.paths_live_saving.add(save_path)
                wandb.save(save_path, policy="live")

    def log_video(self, video_path: str, fps: int, global_step: int) -> None:
        if self.wandb_enabled:
            wandb.log(
                {
                    "videos": wandb.Video(video_path, fps=fps),
                },
                step=global_step,
            )

    def log_text(self, levelno: int, msg: str, global_step: int) -> None:
        print(f"[{logging.getLevelName(levelno)}] {global_step}: {msg}")
