source benchmarks/train_loop.sh

# export WANDB_PROJECT_NAME="rl-algo-impls"
export VIRTUAL_DISPLAY=1

BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-6}"

ALGOS=(
    # "vpg"
    # "dqn"
    "ppo"
)
ENVS=(
    # Basic
    "CartPole-v1"
    "MountainCar-v0"
    "MountainCarContinuous-v0"
    "Acrobot-v1"
    "LunarLander-v2"
    # PyBullet
    "HalfCheetahBulletEnv-v0"
    "AntBulletEnv-v0"
    "HopperBulletEnv-v0"
    "Walker2DBulletEnv-v0"
    # CarRacing
    "CarRacing-v0"
    # Atari
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "QbertNoFrameskip-v4"
)
train_loop "${ALGOS[*]}" "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
