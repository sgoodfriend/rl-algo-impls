import torch

from rl_algo_impls.shared.policy.policy import Policy

class RemotePolicyState:
    def __init__(self, policy: Policy) -> None:
        cpu_device = torch.device("cpu")
        self.state = {k: v.to(cpu_device) for k, v in policy.state_dict().items()}

    def set_on_policy(self, policy: Policy) -> None:
        policy.load_state_dict(self.state)
