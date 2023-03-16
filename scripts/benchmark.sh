while test $# != 0
do
    case "$1" in
        -a) algo=$2 ;;
        -j) n_jobs=$2 ;;
        -p) project_name=$2 ;;
        -s) seeds=$2 ;;
        -e) envs=$2 ;;
        --procgen) procgen=t
    esac
    shift
done

algo="${algo:-ppo}"
n_jobs="${n_jobs:-6}"
project_name="${project_name:-rl-algo-impls-benchmarks}"
seeds="${seeds:-1 2 3}"

BENCHMARK_ENVS=(
    # Basic
    "CartPole-v1"
    "MountainCar-v0"
    "MountainCarContinuous-v0"
    "Acrobot-v1"
    "LunarLander-v2"
    "BipedalWalker-v3"
    # PyBullet
    "HalfCheetahBulletEnv-v0"
    "AntBulletEnv-v0"
    "HopperBulletEnv-v0"
    "Walker2DBulletEnv-v0"
    # CarRacing
    "CarRacing-v0"
    # Atari
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"
    "QbertNoFrameskip-v4"
)
if [ -z $envs ]; then
    echo "-e unspecified; therefore, benchmark training on ${BENCHMARK_ENVS[*]}"
    envs=${BENCHMARK_ENVS[*]}
fi

PROCGEN_ENVS=(
    "procgen-coinrun-easy"
    "procgen-starpilot-easy"
    "procgen-bossfight-easy"
    "procgen-bigfish-easy"
)
if [ "$procgen" = "t" ]; then
    envs=${PROCGEN_ENVS[*]}
fi

bash scripts/train_loop.sh -a $algo -e "$envs" -p $project_name -s "$seeds" # xargs -I CMD -P $n_jobs bash -c CMD
