source benchmarks/train_loop.sh
ALGOS="ppo"
ENVS="PongNoFrameskip-v4 BreakoutNoFrameskip-v4"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-rl-algo-impls-benchmarks}"
BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"
train_loop $ALGOS "$ENVS" $WANDB_PROJECT_NAME | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD