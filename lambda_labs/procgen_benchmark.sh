source benchmarks/train_loop.sh

# export WANDB_PROJECT_NAME="rl-algo-impls"

BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"

ALGOS=(
    # "vpg"
    # "dqn"
    "ppo"
)
ENVS=(
    "procgen-coinrun-v0-easy"
    "procgen-starpilot-v0-easy"
    "procgen-bossfight-v0-easy"
    "procgen-bigfish-v0-easy"
)
train_loop "${ALGOS[*]}" "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
