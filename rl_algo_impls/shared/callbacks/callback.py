from abc import ABC


class Callback(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.timesteps_elapsed = 0

    def on_step(self, timesteps_elapsed: int = 1) -> bool:
        self.timesteps_elapsed += timesteps_elapsed
        return True
