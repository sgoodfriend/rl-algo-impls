ALGO="ppo"
ENVS="HalfCheetah-v4 Ant-v4 Hopper-v4 Walker2d-v4"
BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"
bash scripts/train_loop.sh -a $ALGO -e "$ENVS" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD