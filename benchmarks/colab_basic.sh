source benchmarks/train_loop.sh
ALGOS="ppo"
ENVS="CartPole-v1 MountainCar-v0 MountainCarContinuous-v0 Acrobot-v1 LunarLander-v2"
train_loop $ALGOS "$ENVS" | xargs -I CMD --max-procs=2 bash -c CMD
