# rl-algo-impls

Implementations of reinforcement learning algorithms.

- Competitions
  - [IEEE-CoG2023 MicroRTS
    competition](https://github.com/sgoodfriend/rl-algo-impls/tree/main/rl_algo_impls/microrts):
    Technical details in
    [technical-description.md](https://github.com/sgoodfriend/rl-algo-impls/blob/main/rl_algo_impls/microrts/technical-description.md).
  - [Lux AI Season 2](https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406791)
  - [Lux AI Season 2 - NeurIPS Stage 2](https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/discussion/459891)
- [WandB benchmark reports](https://wandb.ai/sgoodfriend/rl-algo-impls-benchmarks/reportlist)
  - [Basic, MuJoCo, and Atari games
    (v0.0.9)](https://api.wandb.ai/links/sgoodfriend/fdp5mg6h)
    - [v0.0.8](https://api.wandb.ai/links/sgoodfriend/jh3cqbon)
    - [v0.0.4](https://api.wandb.ai/links/sgoodfriend/09frjfcs)
  - [procgen (starpilot, hard)](https://api.wandb.ai/links/sgoodfriend/v1p4976e) and [procgen (easy)](https://api.wandb.ai/links/sgoodfriend/f3w1hwyb): *procgen is no longer supported by this library given its dependence on gym 0.21*
  - [GridNet MicroRTS](https://api.wandb.ai/links/sgoodfriend/zdee7ovm)
  - [MicroRTS Selfplay](https://api.wandb.ai/links/sgoodfriend/5qjlr8ob)
  - [Lux AI Season 2 Training](https://api.wandb.ai/links/sgoodfriend/0yrxywnd)
  - [Lux AI Season 2 NeurIPS Stage 2 Training](https://api.wandb.ai/links/sgoodfriend/ssxupw6m) & [Follow-up](https://api.wandb.ai/links/sgoodfriend/8ozskssn)
- [Huggingface models](https://huggingface.co/models?other=rl-algo-impls)

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
environments [5 basic gymnasium, 4 MuJoCo, CarRacing-v2, and 4 Atari] across 3 seeds) vs
cost ($0.60/hr).

```sh
git clone https://github.com/sgoodfriend/rl-algo-impls.git
cd rl-algo-impls
# git checkout BRANCH_NAME if running on non-main branch
bash ./scripts/setup.sh # End of script will prompt for WandB API key
bash ./scripts/benchmark.sh [-a {"ppo"}] [-e ENVS] [-j {6}] [-p {rl-algo-impls-benchmarks}] [-s {"1 2 3"}]
```

Benchmarking runs are by default upload to a rl-algo-impls-benchmarks project. Runs upload
videos of the running best model and the weights of the best and last model.
Benchmarking runs are tagged with a shorted commit hash (i.e., `benchmark_5598ebc`) and
hostname (i.e., `host_192-9-145-26`)

#### Publishing models to Huggingface

Publishing benchmarks to Huggingface requires logging into Huggingface with a
write-capable API token:

```sh
git config --global credential.helper store
huggingface-cli login
# For example: python benchmark_publish.py --wandb-tags host_192-9-147-166 benchmark_1d4094f --wandb-report-url https://api.wandb.ai/links/sgoodfriend/099h4lvj
# --virtual-display likely must be specified if running on a remote machine.
python benchmark_publish.py --wandb-tags HOST_TAG COMMIT_TAG --wandb-report-url WANDB_REPORT_URL [--virtual-display]
```

#### Hyperparameter tuning with Optuna

Hyperparameter tuning can be done with the `tuning/tuning.sh` script, which runs
multiple processes of optimize.py. Start by doing all the setup meant for training
before running `tuning/tuning.sh`:

```sh
# Setup similar to training above
wandb login
bash scripts/tuning.sh -a ALGO -e ENV -j N_JOBS -s NUM_SEEDS
```

### Google Colab Pro+

3 notebooks in the colab directory are setup to be used with Google Colab:

- [colab_benchmark.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab/colab_benchmark.ipynb):
  Even with a Google Colab Pro+ subscription you'd need to only run parts of the
  benchmark. The file recommends 4 splits (basic+mujoco, carcarcing, atari1, atari2)
  because it would otherwise exceed the 24-hour session limit. This mostly comes from
  being unable to get pool_size above 1 because of WandB errors.
- [colab_train.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab/colab_train.ipynb):
  Train models while being able to specify the env, seeds, and algo. By default training
  runs are uploaded to the rl-algo-impls project.
- [colab_enjoy.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/main/colab/colab_enjoy.ipynb):
  Download models from WandB and evaluate them. Training is likely to be more
  interesting given videos are uploaded.

### macOS

#### Installation

My local development has been on an M1 Mac. These instructions might not be complete,
but these are the approximate setup and usage I've been using:

1. Install libraries with homebrew

```sh
brew install swig
brew install --cask xquartz
brew install pipx
```

2. Download and install Miniconda for arm64

```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

3. Create a conda environment from this repo's
   [environment.yml](https://github.com/sgoodfriend/rl-algo-impls/blob/main/environment.yml)

```sh
conda env create -f environment.yml -n rai_py38_poetry
conda activate rai_py38_poetry
```

4. Install other dependencies with poetry

```sh
pipx install poetry
poetry install -E all
```

#### Usage

Training, benchmarking, and watching the agents playing the environments can be done
locally:

```sh
python train.py [-h] [--algo {ppo}] [--env ENV [ENV ...]] [--seed [SEED ...]] [--wandb-project-name WANDB_PROJECT_NAME] [--wandb-tags [WANDB_TAGS ...]] [--pool-size POOL_SIZE] [-virtual-display]
```

train.py by default uploads to the rl-algo-impls WandB project. Training creates videos
of the running best model, which will cause popups. Creating the first video requires a
display, so you shouldn't shutoff the display until the video of the initial model is
created (1-5 minutes depending on environment). The --virtual-display flag should allow
headless mode, but that hasn't been reliable on macOS.

```sh
python enjoy.py [-h] [--algo {ppo}] [--env ENV] [--seed SEED] [--render RENDER] [--best BEST] [--n_episodes N_EPISODES] [--deterministic-eval DETERMINISTIC_EVAL] [--no-print-returns]
# OR
python enjoy.py [--wandb-run-path WANDB_RUN_PATH]
```

The first enjoy.py where you specify algo, env, and seed loads a model you locally
trained with those parameters and renders the agent playing the environment.

The second enjoy.py downloads the model and hyperparameters from a WandB run. An
example run path is `sgoodfriend/rl-algo-impls-benchmarks/09gea50g`

## Hyperparameters

These are specified in yaml files in the hyperparams directory by game (`atari` is a
special case for all Atari games).

## gym-microrts Setup

```sh
python -m pip install -e '.[microrts]'
```

Requires Java SDK to also be installed.

## Lux AI Season 2 Setup
Lux training uses a [Jux fork](https://github.com/sgoodfriend/jux) that adds support for environments not being in lockstep, stats collection, and other improvements. The fork by default will install the CPU-only version of Jax, which isn't ideal for training, but useful for development. When doing actual training, you'll need an Nvidia GPU and follow these instructions to [install jax[cuda11_cudnn82]==0.4.7](https://github.com/sgoodfriend/jux#install-jax) before installing the jux dependencies:
```sh
pip install --upgrade "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

After installing jax, install the lux dependencies (which includes jux):
```sh
poetry install -E lux
```