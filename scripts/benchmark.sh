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
    esac
    shift
done

algos="${algos:-ppo a2c dqn vpg}"
n_jobs="${n_jobs:-6}"
project_name="${project_name:-rl-algo-impls-benchmarks}"
seeds="${seeds:-1 2 3}"

DISCRETE_ENVS=(
    # Basic
    "CartPole-v1"
    "MountainCar-v0"
    "Acrobot-v1"
    "LunarLander-v2"
    # Atari
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "QbertNoFrameskip-v4"
)
BOX_ENVS=(
    # Basic
    "MountainCarContinuous-v0"
    "BipedalWalker-v3"
    # PyBullet
    "HalfCheetahBulletEnv-v0"
    "AntBulletEnv-v0"
    "HopperBulletEnv-v0"
    "Walker2DBulletEnv-v0"
    # CarRacing
    "CarRacing-v0"
)

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
    elif [ -z "$envs" ]; then
        if [ "$algo" = "dqn" ]; then
            BENCHMARK_ENVS="${DISCRETE_ENVS[*]}"
        else
            BENCHMARK_ENVS="${DISCRETE_ENVS[*]} ${BOX_ENVS[*]}"
        fi
        algo_envs=${BENCHMARK_ENVS[*]}
    fi

    bash scripts/train_loop.sh -a $algo -e "$algo_envs" -p $project_name -s "$seeds" | xargs -I CMD -P $n_jobs bash -c CMD
done