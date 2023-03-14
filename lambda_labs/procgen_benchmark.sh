BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"

ALGO="ppo"
ENVS=(
    "procgen-coinrun-easy"
    "procgen-starpilot-easy"
    "procgen-bossfight-easy"
    "procgen-bigfish-easy"
)
bash benchmarks/train_loop.sh -a $ALGO -e "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
