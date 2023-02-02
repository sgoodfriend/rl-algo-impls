source benchmarks/train_loop.sh
ALGOS="ppo"
ENVS="SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4"
BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"
train_loop $ALGOS "$ENVS" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD