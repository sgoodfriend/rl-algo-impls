while getopts a:e:s:p: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        e) envs=${OPTARG};;
        s) seeds=${OPTARG};;
        p) project_name=${OPTARG};;
    esac
done

WANDB_TAGS=$(bash scripts/tags_benchmark.sh)
project_name="${project_name:-rl-algo-impls-benchmarks}"
seeds="${seeds:-1 2 3}"

train_jobs=""
for env in $(echo $envs); do
    for seed in $seeds; do
        train_jobs+="python train.py --algo $algo --env $env --seed $seed --pool-size 1 --wandb-tags $WANDB_TAGS --wandb-project-name $project_name --virtual-display"$'\n'
    done
done
printf "$train_jobs"