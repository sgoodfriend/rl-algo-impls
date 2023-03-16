ALGO="ppo"
ENVS="HalfCheetahBulletEnv-v0 AntBulletEnv-v0 HopperBulletEnv-v0 Walker2DBulletEnv-v0"
BENCHMARK_MAX_PROCS="${BENCHMARK_MAX_PROCS:-3}"
bash scripts/train_loop.sh -a $ALGO -e "$ENVS" | xargs -I CMD -P $BENCHMARK_MAX_PROCS bash -c CMD