from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
from rl_algo_impls.shared.policy.actor_critic_network.grid2seq_transformer import (
    empty_spaces_mask,
)
from rl_algo_impls.shared.tensor_utils import batch_dict_keys
from rl_algo_impls.wrappers.vector_wrapper import VectorEnv


class OccupancyLinearProbeTrainer:
    def __init__(
        self,
        device: torch.device,
        d_model: int,
        map_shape: Tuple[int, ...],
        learning_rate: float = 3e-4,
        optim_betas: Tuple[float, float] = (0.9, 0.99),
        optim_weight_decay: float = 0.01,
        detach: bool = False,
        residual_layer_idx: int = 0,
    ) -> None:
        self.device = device
        self.probe_w = nn.Parameter(torch.zeros((d_model,) + map_shape)).to(device)
        nn.init.normal_(self.probe_w, mean=0.0, std=0.02)
        self.probe_b = nn.Parameter(torch.zeros(map_shape)).to(device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            [self.probe_w, self.probe_b],
            lr=learning_rate,
            betas=optim_betas,
            weight_decay=optim_weight_decay,
        )
        self.detach = detach
        self.residual_layer_idx = residual_layer_idx

    def train(
        self,
        policy: AbstractPolicy,
        env: VectorEnv,
        steps: int,
        tb_writer: SummaryWriter,
        deterministic_actions: bool = True,
    ) -> None:
        residual_activations = {}

        def hook_fn(module, input, output):
            residual_activations["activation"] = output.x
            residual_activations["key_padding_mask"] = output.key_padding_mask

        assert isinstance(policy, ActorCritic)
        policy.network.backbone.encoders[self.residual_layer_idx].register_forward_hook(
            hook_fn
        )

        obs, _ = env.reset()
        get_action_mask = getattr(env, "get_action_mask", None)

        num_env_steps = steps // env.num_envs + (1 if steps % env.num_envs else 0)
        step = 0
        tqdm_bar = tqdm.tqdm(range(num_env_steps))
        for _ in tqdm_bar:
            action_masks = (
                batch_dict_keys(get_action_mask()) if get_action_mask else None
            )
            act = policy.act(
                obs,
                deterministic=deterministic_actions,
                action_masks=action_masks,
            )

            if self.detach:
                logits = (
                    torch.zeros(
                        residual_activations["activation"].shape[:2]
                        + self.probe_b.shape,
                        dtype=self.probe_b.dtype,
                        device=self.probe_b.device,
                    )
                    + self.probe_b
                )
            else:
                logits = (
                    einops.einsum(
                        residual_activations["activation"],
                        self.probe_w,
                        "b num_ent d_model, d_model h w -> b num_ent h w",
                    )
                    + self.probe_b
                )
            target = torch.tensor(obs[:, 4:6].any(1), device=self.device).float()
            loss = torch.zeros(1, device=self.device)
            num_entities = 0
            num_correct = 0

            keep_mask = ~empty_spaces_mask(
                einops.rearrange(
                    torch.tensor(obs, device=logits.device), "b c h w -> b (h w) c"
                )
            )

            for i in range(logits.size(0)):
                target_labels = target[i]
                if action_masks is not None:
                    entity_mask = action_masks[i][keep_mask[i]].any(-1)
                else:
                    entity_mask = ~residual_activations["key_padding_mask"][i]
                for entity_idx, has_action in enumerate(entity_mask):
                    if not has_action:
                        continue
                    entity_logits = logits[i, entity_idx]
                    loss += self.loss_fn(entity_logits, target_labels)
                    with torch.no_grad():
                        num_correct += (
                            (entity_logits.sigmoid() > 0.5) == target_labels
                        ).float().sum().item() / np.prod(target_labels.shape)
                    num_entities += 1
            if num_entities > 0:
                loss /= num_entities
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            obs, _, _, _, _ = env.step(act)
            step += env.num_envs

            with torch.no_grad():
                tb_writer.add_scalar("loss", loss.item(), global_step=step)
            accuracy = num_correct / num_entities
            tb_writer.add_scalar("accuracy", accuracy, global_step=step)

            tqdm_bar.set_description(
                f"Loss: {loss.item():.4f}, Acc: {accuracy * 100:.1f}%"
            )
