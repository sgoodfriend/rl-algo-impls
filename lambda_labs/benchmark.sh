while getopts a:j:s: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        j) n_jobs=${OPTARG};;
        p) project_name=${OPTARG};;
        s) seeds=${OPTARG};;
    esac
done

n_jobs="${n_jobs:-6}"
project_name="${project_name:-rl-algo-impls-benchmarks}"
seeds="${seeds:-1 2 3}"

ENVS=(
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
bash benchmarks/train_loop.sh -a $algo -e "${ENVS[*]}" -p $project_name -s "$seeds" | xargs -I CMD -P $n_jobs bash -c CMD
