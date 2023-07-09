# MicroRTS

## IEEE-CoG2023 MicroRTS competition

### Agent installation instructions

1. Java (tested 11+) and Python 3.8+ must be installed
2. Download the RAISocketAI archive. For the CoG2023 MicroRTS competition this can be
   downloaded from https://github.com/sgoodfriend/rl-algo-impls/releases/download/v0.0.38/RAISocketAI-0.0.38.zip
3. Unzip the archive: `unzip -j RAISocketAI-0.0.38.zip`
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
[colab_microrts_demo.ipynb](https://github.com/sgoodfriend/rl-algo-impls/blob/80fa237c2ac166efb13d3cb5fff1eb0f52463193/rl_algo_impls/microrts/colab_microrts_demo.ipynb)
in Google Colab (ideally High-RAM Runtime shape, no Hardware accelator).

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
