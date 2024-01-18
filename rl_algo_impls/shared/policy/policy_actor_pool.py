from typing import List, Union

import ray

from rl_algo_impls.shared.policy.policy_actor import PolicyActor


@ray.remote
class PolicyActorPool:
    def __init__(
        self, num_policy_workers: int, cuda_indexes: Union[int, List[int], None]
    ) -> None:
        if isinstance(cuda_indexes, (int, type(None))):
            _cuda_indexes = [cuda_indexes] * num_policy_workers
        elif len(cuda_indexes) == 1:
            _cuda_indexes = cuda_indexes * num_policy_workers
        else:
            _cuda_indexes = cuda_indexes
        assert len(_cuda_indexes) == num_policy_workers

        self.policy_actors = [
            PolicyActor.remote(cuda_idx) for cuda_idx in _cuda_indexes
        ]
        self.consumer_actor_id_to_policy_actor = {}

    def get_policy_for_actor_id(self, actor_id: str) -> PolicyActor:
        if actor_id not in self.consumer_actor_id_to_policy_actor:
            self.consumer_actor_id_to_policy_actor[actor_id] = self.policy_actors[
                len(self.consumer_actor_id_to_policy_actor) % len(self.policy_actors)
            ]
        return self.consumer_actor_id_to_policy_actor[actor_id]

    def get_all_actors(self) -> List[PolicyActor]:
        return self.policy_actors
