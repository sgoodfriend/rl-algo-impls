# export BENCHMARK_MAX_PROCS=1
# export WANDB_PROJECT_NAME="rl-algo-impls"
export VIRTUAL_DISPLAY=1
./benchmarks/colab_basic.sh
./benchmarks/colab_pybullet.sh
./benchmarks/colab_carracing.sh
./benchmarks/colab_atari1.sh
./benchmarks/colab_atari2.sh