# rl-algo-impls

Implementations of reinforcement learning algorithms.

Latest WandB report: https://api.wandb.ai/links/sgoodfriend/6p2sjqtn

## Prerequisites: Weights & Biases (WandB)

Training and benchmarking assumes you have a Weights & Biases project to upload runs to.
By default training goes to a rl-algo-impls project while benchmarks go to
rl-algo-impls-benchmarks. During training and benchmarking runs, videos of the best
models and the model weights are uploaded to WandB.

Before doing anything below, you'll need to create a wandb account and run `wandb
login`.

## Setup and Usage

### Lambda Labs instance for benchmarking

Benchmark runs are uploaded to WandB, which can be made into reports ([for
example](https://api.wandb.ai/links/sgoodfriend/6p2sjqtn)). So far I've found Lambda
Labs A10 instances to be a good balance of performance (14 hours to train PPO in 14
environments [5 basic gym, 4 PyBullet, CarRacing-v0, and 4 Atari] across 3 seeds) vs
cost ($0.60/hr).

```
git clone https://github.com/sgoodfriend/rl-algo-impls.git
cd rl-algo-impls
bash ./lambda_labs/setup.sh
wandb login
bash ./lambda_labs/benchmark.sh
```

Benchmarking runs are by default upload to a rl-algo-impls-benchmarks project. Runs upload
videos of the running best model and the weights of the best and last model.
Benchmarking runs are tagged with a shorted commit hash (i.e., `benchmark_5598ebc`) and
hostname (i.e., `host_192-9-145-26`)

Publishing benchmarks to Huggingface requires logging into Huggingface with a
write-capable API token:

```
git config --global credential.helper store
huggingface-cli login
# For example: python benchmark_publish.py --wandb-tags host_192-9-147-166 benchmark_1d4094f --wandb-report-url https://api.wandb.ai/links/sgoodfriend/099h4lvj
python benchmark_publish.py --wandb-tags HOST_TAG COMMIT_TAG --wandb-report-url WANDB_REPORT_URL
```

### Google Colab Pro+

3 notebooks in the repo are setup to be used with Google Colab:

- [benchmarks/colab_benchmark.ipynb](https://github.com/sgoodfriend/rl-algo-impls/tree/main/benchmarks#:~:text=colab_benchmark.ipynb):
  Even with a Google Colab Pro+ subscription you'd need to only run parts of the
  benchmark. The file recommends 4 splits (basic+pybullet, carcarcing, atari1, atari2)
  because it would otherwise exceed the 24-hour session limit. This mostly comes from
  being unable to get pool_size above 1 because of WandB errors.
- [colab_train.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab_train.ipynb):
  Train models while being able to specify the env, seeds, and algo. By default training
  runs are uploaded to the rl-algo-impls project.
- [colab_enjoy.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab_enjoy.ipynb):
  Download models from WandB and evaluate them. Training is likely to be more
  interesting given videos are uploaded.

### macOS

#### Installation

My local development has been on an M1 Mac. These instructions might not be complete,
but these are the approximate setup and usage I've been using:

1. Install libraries with homebrew

```
brew install swig
brew install --cask xquartz
```

2. Download and install Miniconda for arm64

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

3. Create a conda environment from this repo's
   [environment.yml](https://github.com/sgoodfriend/rl-algo-impls/blob/main/environment.yml)

```
conda env create -f environment.yml -n rl_algo_impls
conda activate rl_algo_impls
```

4. Install other dependencies with poetry

```
poetry install
```

#### Usage

Training, benchmarking, and watching the agents playing the environments can be done
locally:

```
python train.py [-h] [--algo {ppo}] [--env ENV [ENV ...]] [--seed [SEED ...]] [--wandb-project-name WANDB_PROJECT_NAME] [--wandb-tags [WANDB_TAGS ...]] [--pool-size POOL_SIZE]
```

train.py by default uploads to the rl-algo-impls WandB project. Training creates videos
of the running best model, which will cause popups. Creating the first video requires a
display, so you shouldn't shutoff the display until the video of the initial model is
created (1-5 minutes depending on environment).

```
python enjoy.py [-h] [--algo {ppo}] [--env ENV] [--seed SEED] [--render RENDER] [--best BEST] [--n_episodes N_EPISODES] [--deterministic-eval DETERMINISTIC_EVAL] [--no-print-returns]
# OR
python enjoy.py [--wandb-run-path WANDB_RUN_PATH]
```

The first enjoy.py where you specify algo, env, and seed loads a model you locally
trained with those parameters and renders the agent playing the environment.

The second enjoy.py downloads the model and hyperparameters from a WandB run. An
example run path is `sgoodfriend/rl-algo-impls-benchmarks/09gea50g`

```
export WANDB_PROJECT_NAME=rl-algo-impls
# Run any one of these benchmarking scripts normally meant for Google Colab
./benchmarks/colab_basic.sh
./benchmarks/colab_pybullet.sh
./benchmarks/colab_carracing.sh
./benchmarks/colab_atari1.sh
./benchmarks/colab_atari2.sh
```

## Hyperparameters

These are specified in yaml files in the hyperparams directory by game (`atari` is a
special case for all Atari games).
