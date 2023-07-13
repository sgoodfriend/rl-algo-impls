# Technical Description of IEEE-CoG2023 MicroRTS Submission

## Win-Loss Against Prior Competitors on Public Maps

RAISocketAI regularly beets prior competition winners and baselines on 7 of 8 public
maps. The exception is the largest map (64x64). Each cell represents the average result
of RAISocketAI against the opponent AI for 20 matches (10 each as player 1 and player
2). A win is +1, loss is -1, and draw is 0. Even wins and losses would average to a
score of 0. A score of 0.9 corresponds to winning 95% of games (assuming no draws).

| map                     | POWorkerRush | POLightRush | CoacAI | mayari | Map Total |
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
milliseconds) and longest BloodBath (17 milliseconds). The tournament enforced 100 ms
per turn, which no agents ever exceeded.

## Videos Against Mayari (2021 COG winner)


| map                     | Video |
| :---------------------- | :---: |
| basesWorkers8x8A        | <video src="https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/931661d0-003b-4c1a-a3f9-c09c18bfcff9" />  |
| FourBasesWorkers8x8     | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/d7d2469f-8a0f-4007-adc8-800112205e5b |
| NoWhereToRun9x8         | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/cc22ad8f-bd5d-4521-a673-337806c58764 |
| basesWorkers16x16A      | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/72934fdc-5d49-438e-91a3-13b79130fd91 |
| TwoBasesBarracks16x16   | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/5fb19b95-7353-4a03-a09c-ea55a9795eac |
| DoubleGame24x24         | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/d9579fa6-8eb8-4eab-acf7-39b09f0bcd55 |
| BWDistantResources32x32 | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/62f04c88-3d58-43c5-94ab-7705d6abe886 |
| (4)BloodBath.scmB       | https://github.com/sgoodfriend/rl-algo-impls/assets/1751100/0de8632e-a147-403d-b650-486fc3f703b3 |

## References

<a name="Huang2021">[1]</a> Huang, S., Ontañón, S., Bamford, C., & Grela, L. (2021). Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement Learning. arXiv preprint [arXiv:2105.13807](https://arxiv.org/abs/2105.13807).
