import torch


def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # define the mapping of old key names to the new ones
        key_mapping = {
            "network.critic_heads.0.0.0.weight": "network.critic_heads.0.0.weight",
            "network.critic_heads.0.0.0.bias": "network.critic_heads.0.0.bias",
            "network.critic_heads.0.1.0.weight": "network.critic_heads.0.2.weight",
            "network.critic_heads.0.1.0.bias": "network.critic_heads.0.2.bias",
            "network.critic_heads.0.2.0.weight": "network.critic_heads.0.4.weight",
            "network.critic_heads.0.2.0.bias": "network.critic_heads.0.4.bias",
            "network.critic_heads.0.5.weight": "network.critic_heads.0.8.weight",
            "network.critic_heads.0.5.bias": "network.critic_heads.0.8.bias",
            "network.critic_heads.0.7.weight": "network.critic_heads.0.10.weight",
            "network.critic_heads.0.7.bias": "network.critic_heads.0.10.bias",
            "network.critic_heads.1.0.0.weight": "network.critic_heads.1.0.weight",
            "network.critic_heads.1.0.0.bias": "network.critic_heads.1.0.bias",
            "network.critic_heads.1.1.0.weight": "network.critic_heads.1.2.weight",
            "network.critic_heads.1.1.0.bias": "network.critic_heads.1.2.bias",
            "network.critic_heads.1.2.0.weight": "network.critic_heads.1.4.weight",
            "network.critic_heads.1.2.0.bias": "network.critic_heads.1.4.bias",
            "network.critic_heads.1.5.weight": "network.critic_heads.1.8.weight",
            "network.critic_heads.1.5.bias": "network.critic_heads.1.8.bias",
            "network.critic_heads.1.7.weight": "network.critic_heads.1.10.weight",
            "network.critic_heads.1.7.bias": "network.critic_heads.1.10.bias",
            "network.critic_heads.2.0.0.weight": "network.critic_heads.2.0.weight",
            "network.critic_heads.2.0.0.bias": "network.critic_heads.2.0.bias",
            "network.critic_heads.2.1.0.weight": "network.critic_heads.2.2.weight",
            "network.critic_heads.2.1.0.bias": "network.critic_heads.2.2.bias",
            "network.critic_heads.2.2.0.weight": "network.critic_heads.2.4.weight",
            "network.critic_heads.2.2.0.bias": "network.critic_heads.2.4.bias",
            "network.critic_heads.2.5.weight": "network.critic_heads.2.8.weight",
            "network.critic_heads.2.5.bias": "network.critic_heads.2.8.bias",
            "network.critic_heads.2.7.weight": "network.critic_heads.2.10.weight",
            "network.critic_heads.2.7.bias": "network.critic_heads.2.10.bias",
        }
        if k in key_mapping:
            new_state_dict[key_mapping[k]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# Load the state_dict
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))

# Fix the state_dict
fixed_state_dict = fix_state_dict(state_dict)

# Overwrite the old file with the fixed state_dict
torch.save(fixed_state_dict, "model.pth")
