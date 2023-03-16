ALGO="ppo"
ENVS="CartPole-v1 MountainCar-v0 MountainCarContinuous-v0 Acrobot-v1 LunarLander-v2"
BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"
bash scripts/train_loop.sh -a $ALGO -e "$ENVS" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD
