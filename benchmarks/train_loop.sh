train_loop () {
    local WANDB_TAGS="benchmark_$(git rev-parse --short HEAD) host_$(hostname)"
    local algo
    local env
    local seed
    for algo in $(echo $1); do
        for env in $(echo $2); do
            for seed in {1..3}; do
                echo python train.py --algo $algo --env $env --seed $seed --pool-size 1 --wandb-tags $WANDB_TAGS --wandb-project-name $3
            done
        done
    done
}