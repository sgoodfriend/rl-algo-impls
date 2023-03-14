BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-5}"

ALGO="ppo"
ENVS=(
    "procgen-starpilot-hard"
    "procgen-starpilot-hard-2xIMPALA"
    "procgen-starpilot-hard-2xIMPALA-fat"
    "procgen-starpilot-hard-4xIMPALA"
)
bash benchmarks/train_loop.sh -a $ALGO -e "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
