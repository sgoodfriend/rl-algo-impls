from typing import Tuple

import einops
import torch
import torch.nn as nn
import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from rl_algo_impls.shared.policy.abstract_policy import AbstractPolicy
from rl_algo_impls.shared.policy.actor_critic import ActorCritic
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
        policy.network.backbone.encoders[0].register_forward_hook(hook_fn)

        obs, _ = env.reset()
        get_action_mask = getattr(env, "get_action_mask", None)

        num_env_steps = steps // env.num_envs + (1 if steps % env.num_envs else 0)
        step = 0
        tqdm_bar = tqdm.tqdm(range(num_env_steps))
        for _ in tqdm_bar:
            act = policy.act(
                obs,
                deterministic=deterministic_actions,
                action_masks=(
                    batch_dict_keys(get_action_mask()) if get_action_mask else None
                ),
            )

            logits = (
                einops.einsum(
                    residual_activations["activation"],
                    self.probe_w,
                    "b num_ent d_model, d_model h w -> b num_ent h w",
                )
                + self.probe_b
            )
            target = einops.repeat(
                torch.tensor(obs[:, 4:6].any(1), device=self.device).float(),
                "b h w -> b num_ent h w",
                num_ent=logits.size(1),
            )
            loss = torch.zeros(1, device=self.device)
            num_entities = 0
            for i in range(logits.size(0)):
                for entity_idx in range(logits.size(1)):
                    if residual_activations["key_padding_mask"][i, entity_idx]:
                        continue
                    loss += self.loss_fn(logits[i, entity_idx], target[i, entity_idx])
                    num_entities += 1
            loss /= num_entities
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            obs, _, _, _, _ = env.step(act)
            step += env.num_envs

            with torch.no_grad():
                tb_writer.add_scalar("loss", loss.item(), global_step=step)

            tqdm_bar.set_description(f"Loss: {loss.item():.4f}")
