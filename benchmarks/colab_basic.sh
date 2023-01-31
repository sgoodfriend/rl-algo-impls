ALGOS="ppo"
ENVS="CartPole-v1 MountainCar-v0 MountainCarContinuous-v0 Acrobot-v1 LunarLander-v2 HalfCheetahBulletEnv-v0 AntBulletEnv-v0 Walker2DBulletEnv-v0 HopperBulletEnv-v0"
SEEDS="1 2 3"
for algo in $ALGOS; do
    for env in $ENVS; do
        for seed in $SEEDS; do
            python train.py --algo $algo --env $env --seed $seed --pool-size 1
        done
    done
done