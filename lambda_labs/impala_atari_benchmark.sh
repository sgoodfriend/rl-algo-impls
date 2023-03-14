BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-5}"

ALGO="ppo"
ENVS=(
    "impala-PongNoFrameskip-v4"
    "impala-BreakoutNoFrameskip-v4"
    "impala-SpaceInvadersNoFrameskip-v4"
    "impala-QbertNoFrameskip-v4"
    "impala-CarRacing-v0"
)
bash benchmarks/train_loop.sh -a $ALGO -e "${ENVS[*]}" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
