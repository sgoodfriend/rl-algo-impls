# Technical Description of IEEE-CoG2023 MicroRTS Submission

## Win-Loss Against Prior Competitors on Public Maps

RAISocketAI regularly beets prior competition winners and baselines on 7 of 8 public
maps. The exception is the largest map (64x64). Each cell represents the average result
of RAISocketAI against the opponent AI for 20 matches (10 each as player 1 and player
2). A win is +1, loss is -1, and draw is 0. Even wins and losses would average to a
score of 0. A score of 0.9 corresponds to winning 95% of games (assuming no draws).

| map                     | POWorkerRush | POLightRush | CoacAI | Mayari | Map Total |
| :---------------------- | -----------: | ----------: | -----: | -----: | --------: |
| basesWorkers8x8A        |         0.95 |           1 |      1 |      1 |      0.99 |
| FourBasesWorkers8x8     |            1 |           1 |    0.9 |      1 |      0.98 |
| NoWhereToRun9x8         |            1 |           1 |    0.7 |   0.95 |      0.91 |
| basesWorkers16x16A      |            1 |           1 |    0.9 |    0.8 |      0.92 |
| TwoBasesBarracks16x16   |            1 |        0.65 |      1 |      1 |      0.91 |
| DoubleGame24x24         |            1 |        0.95 |    0.9 |      1 |      0.96 |
| BWDistantResources32x32 |            1 |         0.8 |    0.7 |      1 |      0.88 |
| (4)BloodBath.scmB       |          0.9 |          -1 |     -1 |     -1 |     -0.52 |
| AI Total                |         0.98 |        0.68 |   0.64 |   0.72 |      0.75 |

POWorkerRush, POLightRush, and CoacAI use the default AStarPathFinding. The round-robin
tournamnet was run on an Intel Xeon 8358 with PyTorch limited to 8 threads. The avearge
execution time per turn varied by map-size with the shortest being NoWhereToRun9x8 (9
milliseconds) and longest BloodBath (17 milliseconds). The tournament enforces 100 ms
per turn, which no agents ever exceeded.

## Videos Against Mayari (2021 COG winner)

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

## Agent Overview

RAISocketAI is a Java class that communicates with a Python process to determine
actions. RAISocketAI launches the Python process and uses pipes for write and read to
the process.

The Python process loads up 7 different models. During a game, only one model
is used based on the map:

- ppo-Microrts-finetuned-NoWhereToRun-S1-best: NoWhereToRun9x8
- ppo-Microrts-A6000-finetuned-coac-mayari-S1-best: All other maps of size 16x16 and
  smaller
- ppo-Microrts-finetuned-DoubleGame-shaped-S1-best: DoubleGame24x24
- ppo-Microrts-finetuned-DistantResources-shaped-S1-best: BWDistantResources32x32 if
  believed to complete within 75 milliseconds per turn
- ppo-Microrts-squnet-DistantResources-128ch-finetuned-S1-best: BWDistantResources32x32
  if the other model takes over 75 milliseconds per turn
- ppo-Microrts-squnet-map32-128ch-selfplay-S1-best: All other maps where the longest
  dimension is between 17-32
- ppo-Microrts-squnet-map64-64ch-selfplay-S1-best: Maps where the longest dimension is
  over 32

For the tournament above that used public maps on a fast computer, only the first 4
models are used (none of the "squnet" models). The first 4 models (non-squnet) use the
same model architecture. ppo-Microrts-A6000-finetuned-coac-mayari-S1-best (**DoubleCone**) was initially
trained as a base model. The other 3 models were finetuned from the base model training
on their specific maps.

The "DoubleCone" models is a reimplementation of LUX Season 2 4th place
winner's model [[3]](#FLG2023):

1. 4 residual blocks
2. A block with a stride-4 convolution, 6 residual blocks, and 2 stride-2 transpose
   convolutions with a residual connection.
3. 4 residual blocks
4. Actor and critic heads.

Each residual block includes a SqueezeAndExcitation layer.

The "squnet" models are similar to "DoubleCone" in the use of residual blocks with
squeeze-and-excitation and residual connections across convolution-deconvolution blocks.
However, squnet more mimics a U-net architecture in that there are multiple strided
convolution layers to increase receptive field. The table below shows the number of
levels, residual blocks, strides, and other statistics between the DoubleCone and squnet
models:

|                               | DoubleCone                                                                               | squnet-map32         | squnet-map64     |
| ----------------------------- | ---------------------------------------------------------------------------------------- | -------------------- | ---------------- |
| Levels                        | 2                                                                                        | 4                    | 4                |
| Encoder residual blocks/level | [4, 6]                                                                                   | [1, 1, 1, 1]         | [1, 1, 1, 1]     |
| Decoder residual blocks/level | [4]                                                                                      | [1, 1, 1]            | [1, 1, 1]        |
| Stride/level                  | [4]                                                                                      | [2, 2, 4]            | [2, 4, 4]        |
| Deconvolution strides/level   | [[2, 2]<sup>\*</sup>]                                                                    | [2, 2, 4]            | [2, 4, 4]        |
| Channels/level                | [128, 128]                                                                               | [128, 128, 128, 128] | [64, 64, 64, 64] |
| Trainable parameters          | 5,014,865                                                                                | 3,584,657            | 1,420,625        |
| MACs<sup>†</sup>              | 0.70B (16x16)<sup>‡</sup><br>0.40B (12x12)<sup>§</sup><br>1.58B (24x24)<br>2.81B (32x32) | 1.16B (32x32)        | 1.41B (64x64)    |

<sup>\*</sup>2 stride-2 transpose convolutions to match the 1 stride-4 convolution.
<sup>†</sup>Multiply-Accumulates for computing actions for a single observation.
<sup>‡</sup>All maps smaller than 16x16 (except NoWhereToRun9x8) are padded with walls up
to 16x16.
<sup>§</sup>NoWhereToRun9x8 is padded with walls up to 12x12.

All models have one actor head, which outputs an action for every location
(GridNet in [[1]](#Huang2021Gym)). All models have 3 value heads for 3 different value
functions:

1. Dense reward similar to [[1]](#Huang2021Gym), except reward for building combat units
   is split by combat unit type scaled by build-time. Linear activation.
2. Win-loss sparse reward ranging from +1 for win and -1 for loss. Tanh activation.
3. Cost-based unit score similar to [[4]](#Clemens2021)

These 3 value heads are used to mix-and-match rewards over the course of training,
generally starting with dense rewards using 1 and 3 and finishing with only win-loss
sparse rewards by the end.

## References

<a name="Huang2021Gym">[1]</a> Huang, S., Ontañón, S., Bamford, C., & Grela, L. (2021).
Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep
Reinforcement Learning. arXiv preprint
[arXiv:2105.13807](https://arxiv.org/abs/2105.13807).

<a name="Huang2021Generalize">[2]</a> Huang, S., & Ontañón, S. (2021). Measuring Generalization of Deep Reinforcement Learning Applied to Real-time Strategy Games. In Proceedings of the AAAI 2021 Reinforcement Learning in Games Workshop. Retrieved from http://aaai-rlg.mlanctot.info/papers/AAAI21-RLG_paper_33.pdf

<a name="FLG2023">[3]</a> FLG. (2023). FLG's Approach - Deep Reinforcement Learning with a Focus on Performance - 4th place. Kaggle. Retrieved from https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406702

<a name="Clemens2021">[4]</a> Winter, C. (2021, March 24). Mastering Real-Time Strategy Games with Deep Reinforcement Learning: Mere Mortal Edition. Clemens' Blog. Retrieved from https://clemenswinter.com/2021/03/24/mastering-real-time-strategy-games-with-deep-reinforcement-learning-mere-mortal-edition/
