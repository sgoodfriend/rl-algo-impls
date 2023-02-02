train_loop () {
    local WANDB_TAGS="benchmark_$(git rev-parse --short HEAD) host_$(hostname)"
    local algo
    local env
    local seed
    local WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-rl-algo-impls-benchmarks}"
    local args=()
    (( VIRTUAL_DISPLAY == 1)) && args+=("--virtual-display")
    for algo in $(echo $1); do
        for env in $(echo $2); do
            for seed in {1..3}; do
                echo python train.py --algo $algo --env $env --seed $seed --pool-size 1 --wandb-tags $WANDB_TAGS --wandb-project-name $WANDB_PROJECT_NAME ${args[@]}
            done
        done
    done
}