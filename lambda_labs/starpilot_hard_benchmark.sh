source benchmarks/train_loop.sh

# export WANDB_PROJECT_NAME="rl-algo-impls"

BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-1}"

ALGOS=(
    "ppo"
)
ENVS=(
    "procgen-starpilot-hard"
    "procgen-starpilot-hard-2xIMPALA"
    "procgen-starpilot-hard-2xIMPALA-fat"
    "procgen-starpilot-hard-4xIMPALA"
)
train_loop "${ALGOS[*]}" "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
