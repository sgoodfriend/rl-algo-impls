source benchmarks/train_loop.sh

# export WANDB_PROJECT_NAME="rl-algo-impls"

BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"

ALGOS=(
    # "vpg"
    # "dqn"
    "ppo"
)
ENVS=(
    "procgen-coinrun-easy"
    "procgen-starpilot-easy"
    "procgen-bossfight-easy"
    "procgen-bigfish-easy"
)
train_loop "${ALGOS[*]}" "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
