import os
from typing import Dict, Type, TypeVar

from rl_algo_impls.shared.algorithm import Algorithm
from rl_algo_impls.shared.policy.policy import Policy
from rl_algo_impls.shared.trackable import Trackable

T = TypeVar("T", bound=Trackable)


class AgentState:
    def __init__(self) -> None:
        self.load_path = None
        self._registered: Dict[Type[Trackable], Trackable] = {}

    @property
    def policy(self) -> Policy:
        return self.get_registered(Policy)

    @policy.setter
    def policy(self, policy: Policy) -> None:
        self._register(policy, Policy)

    @property
    def algo(self) -> Algorithm:
        return self.get_registered(Algorithm)

    @algo.setter
    def algo(self, algo: Algorithm) -> None:
        self._register(algo, Algorithm)

    def register(self, trackable: Trackable) -> None:
        self._register(trackable, type(trackable))

    def _register(self, trackable: Trackable, cls: Type[Trackable]) -> None:
        if cls in self._registered:
            trackable.sync(self._registered[cls])
        else:
            self._registered[cls] = trackable
        if self.load_path is not None:
            trackable.load(self.load_path)

    def get_registered(self, trackable_type: Type[T]) -> T:
        registered = self._registered[trackable_type]
        assert isinstance(registered, trackable_type)
        return registered

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for trackable in self._registered.values():
            trackable.save(path)

    def load(self, path: str) -> None:
        self.load_path = path
        for trackable in self._registered.values():
            trackable.load(path)
