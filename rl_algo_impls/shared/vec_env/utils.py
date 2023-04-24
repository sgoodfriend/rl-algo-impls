import gym

from rl_algo_impls.runner.config import Config


def import_for_env_id(env_id: str) -> None:
    if "BulletEnv" in env_id:
        import pybullet_envs
    if "Microrts" in env_id:
        import gym_microrts
    if "LuxAI_S2" in env_id:
        import luxai_s2


def is_atari(config: Config) -> bool:
    spec = gym.spec(config.env_id)
    return "AtariEnv" in str(spec.entry_point)


def is_bullet_env(config: Config) -> bool:
    return "BulletEnv" in config.env_id


def is_car_racing(config: Config) -> bool:
    return "CarRacing" in config.env_id


def is_gym_procgen(config: Config) -> bool:
    return "procgen" in config.env_id


def is_microrts(config: Config) -> bool:
    return "Microrts" in config.env_id


def is_lux(config: Config) -> bool:
    return "LuxAI_S2" in config.env_id
