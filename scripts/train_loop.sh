while getopts a:e:s:p:d: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        e) envs=${OPTARG};;
        s) seeds=${OPTARG};;
        p) project_name=${OPTARG};;
        d) devices=${OPTARG};;
    esac
done

WANDB_TAGS=$(bash scripts/tags_benchmark.sh)
project_name="${project_name:-rl-algo-impls-benchmarks}"
seeds="${seeds:-1 2 3}"
devices="${devices:-1}"

train_jobs=""
job_idx=0
for env in $(echo $envs); do
    for seed in $seeds; do
        cmd="poetry run python train.py --algo $algo --env $env --seed $seed --pool-size 1 --wandb-tags $WANDB_TAGS --wandb-project-name $project_name --virtual-display"
        if [ "$devices" -gt 1 ] && [ "$job_idx" -lt "$devices" ]; then
            cmd+=" --device-index $job_idx"
        fi
        train_jobs+="$cmd"$'\n'
        job_idx=$((job_idx+1))
    done
done
printf "$train_jobs"