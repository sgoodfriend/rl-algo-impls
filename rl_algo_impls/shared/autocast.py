import torch


from contextlib import contextmanager


@contextmanager
def maybe_autocast(autocast_enabled: bool, device: torch.device):
    if autocast_enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            yield
    else:
        yield