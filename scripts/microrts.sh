while test $# != 0
do
    case "$1" in
        -a) algos=$2 ;;
        -j) n_jobs=$2 ;;
        -p) project_name=$2 ;;
        -s) seeds=$2 ;;
        -e) envs=$2 ;;
        --procgen) procgen=t ;;
        --microrts) microrts=t ;;
        --no-mask-microrts) no_mask_microrts=t ;;
        --microrts-ai) microrts_ai=t ;;
    esac
    shift
done

algos="${algos:-ppo}"
n_jobs="${n_jobs:-1}"
project_name="${project_name:-mech-interp-rl-algo-impls}"
seeds="${seeds:-1}"

BASIC_ENVS=(
    "CartPole-v1"
    "MountainCar-v0"
    "Acrobot-v1"
    "LunarLander-v2"
)
ATARI_ENVS=(
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "QbertNoFrameskip-v4"
)
BOX_ENVS=(
    # Basic
    "MountainCarContinuous-v0"
    "BipedalWalker-v3"
    # MuJoCo
    "HalfCheetah-v4"
    "Ant-v4"
    "Hopper-v4"
    "Walker2d-v4"
    # CarRacing
    "CarRacing-v2"
)

train_jobs=""
for algo in $(echo $algos); do
    if [ "$procgen" = "t" ]; then
        PROCGEN_ENVS=(
            "procgen-coinrun-easy"
            "procgen-starpilot-easy"
            "procgen-bossfight-easy"
            "procgen-bigfish-easy"
        )
        algo_envs=${PROCGEN_ENVS[*]}
    elif [ "$microrts" = "t" ]; then
        MICRORTS_ENVS=(
            "MicrortsMining-v1"
            "MicrortsAttackShapedReward-v1"
            "MicrortsRandomEnemyShapedReward3-v1"
        )
        algo_envs=${MICRORTS_ENVS[*]}
    elif [ "$no_mask_microrts" = "t" ]; then
        NO_MASK_MICRORTS_ENVS=(
            "MicrortsMining-v1-NoMask"
            "MicrortsAttackShapedReward-v1-NoMask"
            "MicrortsRandomEnemyShapedReward3-v1-NoMask"
        )
        algo_envs=${NO_MASK_MICRORTS_ENVS[*]}
    elif [ "$microrts_ai" == "t" ]; then
        MICRORTS_AI_ENVS=(
            "MicrortsDefeatCoacAIShaped-v3"
            "MicrortsDefeatCoacAIShaped-v3-diverseBots"
        )
        algo_envs=${MICRORTS_AI_ENVS[*]}
    elif [ -z "$envs" ]; then
        if [ "$algo" = "dqn" ]; then
            BENCHMARK_ENVS="${BASIC_ENVS[*]} ${ATARI_ENVS[*]}"
        else
            BENCHMARK_ENVS="${BASIC_ENVS[*]} ${BOX_ENVS[*]} ${ATARI_ENVS[*]}"
        fi
        algo_envs=${BENCHMARK_ENVS[*]}
    else
        algo_envs=$envs
    fi

    train_jobs+=$(bash scripts/train_loop.sh -a $algo -e "$algo_envs" -p $project_name -s "${seeds[*]}")$'\n'
done

printf "$train_jobs" | xargs -I CMD -P $n_jobs bash -c CMD
