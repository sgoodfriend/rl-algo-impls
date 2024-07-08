# MicroRTS

## Training
Training requires an Nvidia GPU with CUDA support. It's fine to develop on a machine without a GPU, but training will likely be so slow as to be impractical.

1. Install Java JDK (Ubuntu 20.04's default JDK is 11, which is fine) and swig
2. Install [poetry](https://python-poetry.org/docs/#installation).
3. Install [git-lfs](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing) and `git lfs pull` to download the jar files.
4. Install rl_algo_impls with `poetry install -E microrts`.
5. Run `poetry run wandb login` and follow the instructions to add the Weights & Biases API key to the environment.
6. Train using `./scripts/microrts.sh -a {[ppo], appo, dppo} -e {NAME FROM ENVIRONMENT YAML}`.

For example, my setup for using APPO to train `Microrts-small-net-bw8a-40m-ent5-lr3c-05wb2lwr-vf50-nga-a10` on a Lambda Cloud A10 instance is:
```sh
git clone https://github.com/sgoodfriend/rl-algo-impls.git
cd rl-algo-impls
# git checkout BRANCH_NAME if running on non-main branch
./scripts/setup.sh --microrts # Handles steps 1-5
./scripts/microrts.sh -a appo -e Microrts-small-net-bw8a-40m-ent5-lr3c-05wb2lwr-vf50-nga-a10
```

## IEEE-CoG2024 MicroRTS competition

Note: This submission is RAI-BC-PPO from [A Competition Winning Deep Reinforcement
Learning Agent in microRTS](https://arxiv.org/abs/2402.08112), which is not optimized
for competition run-time limits. It will likely time-out on the largest (64x64) maps.

1. Java (tested 11+), swig, and Python 3.8-3.10 (ideally 3.10) must be installed
2. Download the RAISocketAI archive. For the CoG2023 MicroRTS competition this can be
   downloaded from https://github.com/sgoodfriend/rl-algo-impls/releases/download/v0.2.0/rl_algo_impls-0.2.0.zip
3. Unzip the archive: `unzip -j rl_algo_impls-0.2.0.zip`
4. Upgrade and install Python depdendencies:

```
python -m pip install --upgrade pip
python -m pip install --upgrade torch
```

5. Install the `.whl` file:

```
python -m pip install --upgrade rl_algo_impls-0.2.0-py3-none-any.whl
```

The above steps makes `rai_microrts` callable within the terminal. `RAIBCPPO.java`
uses this to start a Python child process, which is used to compute actions.
`RAISocketAI.java` runs the 2023 competition winning model.

To see this demonstrated in Google Colab running a small command-line tournament, open
[colab_microrts_demo.ipynb](colab_microrts_demo.ipynb)
in Google Colab (ideally High-RAM Runtime shape, no Hardware accelator).


## IEEE-CoG2023 MicroRTS competition

_Technical details in [technical-description.md](https://github.com/sgoodfriend/rl-algo-impls/blob/main/rl_algo_impls/microrts/technical-description.md)_

### Agent installation instructions

1. Java (tested 11+) and Python 3.8+ must be installed
2. Download the RAISocketAI archive. For the CoG2023 MicroRTS competition this can be
   downloaded from https://github.com/sgoodfriend/rl-algo-impls/releases/download/v0.0.38-bugfix/RAISocketAI-0.0.38-bugfix.zip
3. Unzip the archive: `unzip -j RAISocketAI-0.0.38-bugfix.zip`
4. Upgrade and install Python depdendencies:

```
python -m pip install --upgrade pip
python -m pip install setuptools==65.5.0 wheel==0.38.4
python -m pip install --upgrade torch
```

5. Install the `.whl` file:

```
python -m pip install --upgrade rl_algo_impls-0.0.38-py3-none-any.whl
```

The above steps makes `rai_microrts` callable within the terminal. `RAISocketAI.java`
uses this to start a Python child process, which is used to compute actions.

To see this demonstrated in Google Colab running a small command-line tournament, open
[colab_microrts_demo.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/v0.0.38-bugfix/rl_algo_impls/microrts/colab_microrts_demo.ipynb)
in Google Colab (ideally High-RAM Runtime shape, no Hardware accelator).

### Win-Loss Against Prior Competitors on Public Maps

RAISocketAI regularly beats prior competition winners and baselines on 7 of 8
competition public maps. The exception is the largest map (64x64). Each cell below represents the average result
of RAISocketAI against the opponent AI for 100 matches (50 each as player 1 and player
2). A win is +1, loss is -1, and draw is 0. Same number of wins and losses would average to a
score of 0. A score of 0.9 corresponds to winning 95% of games (assuming no draws).

| map                     | POWorkerRush | POLightRush | CoacAI | Mayari | Map Total |
| :---------------------- | -----------: | ----------: | -----: | -----: | --------: |
| basesWorkers8x8A        |         0.91 |           1 |   0.98 |      1 |      0.97 |
| FourBasesWorkers8x8     |            1 |           1 |      1 |   0.97 |      0.99 |
| NoWhereToRun9x8         |            1 |           1 |   0.93 |   0.97 |      0.98 |
| basesWorkers16x16A      |            1 |           1 |   0.78 |   0.97 |      0.94 |
| TwoBasesBarracks16x16   |            1 |        0.78 |   0.98 |      1 |      0.94 |
| DoubleGame24x24         |            1 |           1 |   0.85 |      1 |      0.96 |
| BWDistantResources32x32 |            1 |        0.84 |   0.82 |   0.97 |      0.91 |
| (4)BloodBath.scmB       |         0.96 |          -1 |     -1 |     -1 |     -0.51 |
| AI Total                |         0.98 |         0.7 |   0.67 |   0.74 |      0.77 |

Mayari was the 2021 COG winner (prior competition), and CoacAI was the 2020 COG winner. POWorkerRush
and POLightRush are baseline AIs. POWorkerRush, POLightRush, and CoacAI use the default AStarPathFinding.

The round-robin tournamnet was run on a 2018 Mac Mini with Intel i7-8700B CPU (6-core,
3.2 GHz) with PyTorch limited to 6 threads. The avearge
execution time per turn varied by map-size with the shortest being NoWhereToRun9x8 (9
milliseconds) and longest BWDistantResources32x32 and BloodBath (22 ms). The tournament enforces 100 ms
per turn. RAISocketAI exceeded the 100 ms limit and needed to skip its turn on less than
0.001% of turns but did lose by timeout in 5 matches (4 BloodBath [1% of BloodBath
matches], 1 BWDistantResources [0.25%])

### Videos Against Mayari (2021 COG winner)

In the videos below, RAISocketAI is the blue player (units outlined in blue, opponent
units outlined in red). RAISocketAI wins in all the videos, except (4)BloodBath.scmB.
RAISocketAI is the left or up player, except in (4)BloodBath.scmB (down-right).

| map                     |                                                      Video                                                       |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------: |
| basesWorkers8x8A        | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/931661d0-003b-4c1a-a3f9-c09c18bfcff9" /> |
| FourBasesWorkers8x8     | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/d7d2469f-8a0f-4007-adc8-800112205e5b" /> |
| NoWhereToRun9x8         | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/cc22ad8f-bd5d-4521-a673-337806c58764" /> |
| basesWorkers16x16A      | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/72934fdc-5d49-438e-91a3-13b79130fd91" /> |
| TwoBasesBarracks16x16   | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/5fb19b95-7353-4a03-a09c-ea55a9795eac" /> |
| DoubleGame24x24         | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/d9579fa6-8eb8-4eab-acf7-39b09f0bcd55" /> |
| BWDistantResources32x32 | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/62f04c88-3d58-43c5-94ab-7705d6abe886" /> |
| (4)BloodBath.scmB       | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/0de8632e-a147-403d-b650-486fc3f703b3" /> |

### v0.0.38 vs v0.0.38-bugfix

The bugfix build only has a change in the Java-side RAISocketAI where the thread pool is
allowed to empty to 0 threads. This fixes an issue where scripts would not terminate
because of threads awaiting tasks in RAISocketAI. The bug does not affect gameplay at all.
The fix only changes the program end behavior. The Python `.whl` file is exactly the same
between v0.0.38 and v0.0.38-bugfix.

If you want to continue using
[v0.0.38](https://github.com/sgoodfriend/rl-algo-impls/releases/download/v0.0.38/RAISocketAI-0.0.38.zip)
and want to fix the script termination behavior, add `System.exit(0);` at the end of the
`main` function.

### Best models variant

`RAISocketAIBestModels.java` is a subclass of `RAISocketAI`, which always uses the best
model for the given map. This still respects the `timeBudget` passed into it, so if the
model takes over the `timeBudget` it'll instead return an empty PlayerAction.

`RAISocketAI` has 3 "general" models for 3 different sets of map sizes:

1. Maps whose longest dimension is 16
2. Maps whose longest dimension is 17-32
3. Maps whose longest dimension is 33-64

Additionally, there are models finetuned on specific maps:

1. maps/NoWhereToRun9x8.xml
2. maps/DoubleGame24x24.xml
3. maps/BWDistantResources32x32.xml (2 models: one larger, the other faster)

During the `preGameAnalysis` step, `RAISocketAI` computes the next action to both warm
up data structures and to estimate how long each model is likely to take.
It will pick the largest model that will likely take less than 75% of the `timeBudget`,
which on slower machines will be the smaller, faster model. `RAISocketAIBestModels` will
always pick the larger model.

If you want to run the agent with the best models, either make sure the machine is fast
enough to generally complete turns in under 50 milliseconds (Apple M1 Max, Intel Core
i7-8700B, Intel Xeon 8358 are all examples that are easily fast enough) OR increase the
timeBudget sufficiently high. `RAISocketAIBestModels` shouldn't be necessary, but can be
used to ensure it's always used.
