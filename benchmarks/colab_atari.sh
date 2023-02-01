source benchmarks/train_loop.sh
ALGOS="ppo"
ENVS="PongNoFrameskip-v4 BreakoutNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 QbertNoFrameskip-v4"
train_loop $ALGOS "$ENVS" | xargs -I CMD --max_procs=3 bash -c CMD