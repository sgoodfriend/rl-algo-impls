source benchmarks/train_loop.sh

# export WANDB_PROJECT_NAME="rl-algo-impls"

BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-5}"

ALGOS=(
    # "vpg"
    # "dqn"
    "ppo"
)
ENVS=(
    "impala-PongNoFrameskip-v4"
    "impala-BreakoutNoFrameskip-v4"
    "impala-SpaceInvadersNoFrameskip-v4"
    "impala-QbertNoFrameskip-v4"
    "impala-CarRacing-v0"
)
train_loop "${ALGOS[*]}" "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
