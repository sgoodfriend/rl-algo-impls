source benchmarks/train_loop.sh
ALGOS="ppo"
ENVS="CarRacing-v0"
train_loop $ALGOS "$ENVS" | xargs -I CMD --max_procs=3 bash -c CMD