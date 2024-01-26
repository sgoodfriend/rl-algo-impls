from collections import defaultdict
from typing import DefaultDict, List, Optional, Union

import ray

from rl_algo_impls.shared.policy.policy_actor import PolicyActor


@ray.remote
class PolicyActorPool:
    def __init__(
        self,
        num_policy_workers: int,
        device_type: str,
        cuda_indexes: Union[int, List[int], None],
    ) -> None:
        if isinstance(cuda_indexes, (int, type(None))):
            _cuda_indexes = [cuda_indexes] * num_policy_workers
        else:
            assert (
                num_policy_workers % len(cuda_indexes) == 0
            ), f"cuda_indexes must be multiple of num_policy_workers"
            _cuda_indexes = cuda_indexes * (num_policy_workers // len(cuda_indexes))

        assert len(_cuda_indexes) == num_policy_workers

        self.policy_actors = [
            PolicyActor.remote(device_type, cuda_idx) for cuda_idx in _cuda_indexes
        ]
        self.policy_actors_by_cuda_index = {}
        for cuda_index, policy_actor in zip(_cuda_indexes, self.policy_actors):
            if cuda_index not in self.policy_actors_by_cuda_index:
                self.policy_actors_by_cuda_index[cuda_index] = []
            self.policy_actors_by_cuda_index[cuda_index].append(policy_actor)

        self.consumer_actor_id_to_policy_actor = {}
        self.num_actors_by_policy_actor_id = {
            get_actor_id(policy_actor): 0 for policy_actor in self.policy_actors
        }

    def get_policy_for_actor_id(
        self, actor_id: str, cuda_index: Optional[int]
    ) -> PolicyActor:
        if actor_id not in self.consumer_actor_id_to_policy_actor:
            policy_actors = self.policy_actors_by_cuda_index.get(
                cuda_index, self.policy_actors
            )
            fewest_actor_sorted = sorted(
                [
                    (
                        self.num_actors_by_policy_actor_id[get_actor_id(policy_actor)],
                        idx,
                    )
                    for idx, policy_actor in enumerate(policy_actors)
                ]
            )
            policy_actor = policy_actors[fewest_actor_sorted[0][1]]
            self.consumer_actor_id_to_policy_actor[actor_id] = policy_actor
            self.num_actors_by_policy_actor_id[get_actor_id(policy_actor)] += 1
        return self.consumer_actor_id_to_policy_actor[actor_id]

    def get_all_actors(self) -> List[PolicyActor]:
        return self.policy_actors


def get_actor_id(actor) -> str:
    return actor._actor_id.hex()
