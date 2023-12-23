from typing import Optional

from rl_algo_impls.checkpoints.checkpoints_manager import PolicyCheckpointsManager
from rl_algo_impls.rollout.rollout import RolloutGenerator
from rl_algo_impls.runner.config import Config, EnvHyperparams
from rl_algo_impls.shared.agent_state import AgentState
from rl_algo_impls.shared.callbacks.summary_wrapper import SummaryWrapper
from rl_algo_impls.shared.vec_env.env_spaces import EnvSpaces
from rl_algo_impls.shared.vec_env.make_env import make_env


class InProcessRolloutGenerator(RolloutGenerator):
    def __init__(
        self,
        config: Config,
        agent_state: AgentState,
        tb_writer: SummaryWrapper,
        checkpoints_wrapper: Optional[PolicyCheckpointsManager],
        **kwargs
    ) -> None:
        super().__init__(agent_state)
        self.vec_env = make_env(
            config,
            EnvHyperparams(**config.env_hyperparams),
            agent_state,
            tb_writer=tb_writer,
            checkpoints_manager=checkpoints_wrapper,
        )
        self._env_spaces = EnvSpaces.from_vec_env(self.vec_env)

    @property
    def env_spaces(self) -> EnvSpaces:
        return self._env_spaces
