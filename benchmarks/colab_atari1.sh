ALGOS="ppo"
ENVS="PongNoFrameskip-v4 BreakoutNoFrameskip-v4"
SEEDS="1 2 3"
WANDB_TAGS="benchmark_$(git rev-parse --short HEAD) host_$(hostname)"
for algo in $ALGOS; do
    for env in $ENVS; do
        for seed in $SEEDS; do
            python train.py --algo $algo --env $env --seed $seed --pool-size 1 --wandb-tags $WANDB_TAGS
        done
    done
done