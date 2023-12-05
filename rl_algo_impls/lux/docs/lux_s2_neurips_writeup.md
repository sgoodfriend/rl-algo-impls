# PPO using Jux for environment vectorization

Kaggle code submission:
[https://www.kaggle.com/code/sgoodfriend/lux2-neurips-stage-2](https://www.kaggle.com/code/sgoodfriend/lux2-neurips-stage-2)

Training repo:
[https://github.com/sgoodfriend/rl-algo-impls-lux-nips1/commit/30800ac](https://github.com/sgoodfriend/rl-algo-impls-lux-nips1/commit/30800ac)

JUX repo: [https://github.com/sgoodfriend/jux](https://github.com/sgoodfriend/jux):
biggest changes are to support environments not being in lockstep, stats collection,
and allowing for adjacent factories (for 16x16 map training).

Weights & Biases report: [Lux S2 NeurIPS Training Report](https://wandb.ai/sgoodfriend/rl-algo-impls-lux-nips1/reports/Lux-S2-NeurIPS-Training-Report--Vmlldzo2MTMyODc3?accessToken=a8xwpu4xi7zavhwmyavxt5lbiejk6wjn1o2eh3v8c3lc416bo11oatirp2pxlzet)

## Environment
Jux allows training with vectorized environments. I used 1024 environments for training
on 16x16 and 32x32 maps and 512 environments for training on 64x64 maps. I'm using a fork of [Jux](https://github.com/sgoodfriend/jux) for training. The fork has
the following changes and extensions:
- Fix incorrectly computing valid_spawns_mask
- EnvConfig option to support adjacent factory spawns (default off). I use this for
  16x16 map training because the default requirement of 6 spaces away could push
  factories too far away from resources.
- Reward -1000 if player lost game for no factories (mimics Lux)
- step_unified combines step_factory_placement and step_late_game
- Environments don't need to run in lockstep and can have different numbers of factories
  to place (externally replace individual envs with new ones when they finish)
- Stats collection (generation [resources, bots, kills], resources [lichen, bots, factories], actions)

I convert the Jux observation to a Gridnet observation with the following observation
features for every position:

|                                | Range       | Comments                                                                         |
| ------------------------------ | ----------- | -------------------------------------------------------------------------------- |
| x                              | [-1, 1]     |                                                                                  |
| y                              | [-1, 1]     |                                                                                  |
| ice                            | [0, 1]      | 1 if ice on tile                                                                 |
| ore                            | [0, 1]      | 1 if ore on tile                                                                 |
| non-zero-rubble                | [0, 1]      | 1 if rubble > 0                                                                  |
| rubble                         | [0, 1]      | Linearly maps [0, 100]                                                           |
| lichen                         | [0, 1]      | Linearly maps [0, 100]. Either owner.                                            |
| lichen at one                  | [0, 1]      | 1 iff lichen 1 on tile                                                           |
| own lichen                     | [0, 1]      | 1 if lichen on tile owned by self                                                |
| opponent lichen                | [0, 1]      | 1 if lichen on tile opponent owned                                               |
| game progress                  | [-0.021, 1) | Maps real_env_steps [-21, 1000)                                                  |
| day cycle                      | (-0.6, 1]   | Starts at 1, 0 at dusk, negative in night. Resets to 1 at dawn                   |
| factories to place             | [0, 1]      | 1 if all factories to place, 0 when none                                         |
| own factory                    | [0, 1]      | 1 if own factory (only center tile)                                              |
| opponent factory               | [0, 1]      | 1 if opponent factory (only center tile)                                         |
| ice-water factory              | [0, 1)      | Ice/4+Water. Exponential decay function. λ = 1/50                                |
| water cost                     | [0, 1)      | Cost to water lichen (factory tiles and owned lichen). EDF λ = 1/10              |
| own factory tile               | [0, 1]      | 1 if own factory tile (9 tiles per factory)                                      |
| opponent factory tile          | [0, 1]      | 1 if opponent factory tile (9 tiles per factory)                                 |
| own unit                       | [0, 1]      |                                                                                  |
| opponent unit                  | [0, 1]      |                                                                                  |
| unit is heavy                  | [0, 1]      | 1 if heavy own or opponent unit                                                  |
| ice factory                    | [0, 1)      | Exponential decay function. λ = 1/500                                            |
| ore factory                    | [0, 1)      | Exponential decay function. λ = 1/1000                                           |
| water factory                  | [0, 1)      | Exponential decay function. λ = 1/1000                                           |
| metal factory                  | [0, 1)      | Exponential decay function. λ = 1/100                                            |
| power factory                  | [0, 1)      | Exponential decay function. λ = 1/3000                                           |
| ice unit                       | 4×[0, 1]    | Fraction of heavy capacity, fraction of capacity, at light capacity, at capacity |
| ore unit                       | 4×[0, 1]    | "                                                                                |
| water unit                     | 4×[0, 1]    | "                                                                                |
| metal unit                     | 4×[0, 1]    | "                                                                                |
| power unit                     | 4×[0, 1]    | "                                                                                |
| enqueued action                | 24×[0, 1]   | 1 if action (and subactions) are enqueued                                        |
| own unit could be in direction | 4×[0, 1]    | 1 if other unit could be in direction next step (N, E, S, W)                     |

I take care of computing amounts of resources in my action handling logic. The model
only handles position for factory placement while I assign the initial water per factory
(150) and enough metal for 1 or 2 heavy units (100 or 200) or 150 if not possible. For
example, for 1 to 4 factories to place:
| Factories to place | Metal |     |     |     |
| ------------------ | ----- | --- | --- | --- |
| 1                  | 150   |     |     |     |
| 2                  | 200   | 100 |     |     |
| 3                  | 200   | 150 | 100 |     |
| 4                  | 200   | 200 | 100 | 100 |

I only allow factories to be placed on tiles that would be adjacent to ice OR ore. I
allow factories to be placed adjacent to ore but not ice to help the model learn to mine
ore and build robots.

I split direction and resources between the action subtypes, resulting in the following
action space per position:
|                    | 0          | 1           | 2           | 3            | 4             | 5        |
| ------------------ | ---------- | ----------- | ----------- | ------------ | ------------- | -------- |
| Factory Action     | do nothing | build light | build heavy | water lichen |               |          |
| Unit Action Type   | move       | transfer    | pickup      | dig          | self-destruct | recharge |
| Move Direction     | north      | east        | south       | west         |               |          |
| Transfer Direction | north      | east        | south       | west         |               |          |
| Transfer Resource  | ice        | ore         | water       | metal        | power         |          |
| Pickup Resource    | ice        | ore         | water       | metal        | power         |

I heavily used invalid action masking to both eliminate no-op actions (e.g. actions on
non-own unit or factory positions, moves or transfers off map or opponent factory, or invalid actions
because insufficient power or resources) and ill-advised actions:
- Don't water lichen if it would result in water being less than the number of game
  steps remaining.
- Don't transfer resources off factory tiles.
  - Exception: Allow transferring power to a unit from a factory tile if the destination
    unit has been digging.
- Cannot pickup resources other than power
  - Exception: Light robots can pickup water if the factory has sufficient water.
- Only allow digging on resources, opponent lichen, and rubble that is adjacent to a
  factory's lichen grow area (prevents digging on distant rubble).
- Only allow moving in a rectangle containing all resources, diggable areas (see above),
  own units, and opponent lichen.
- Only lights can self-destruct and only if they are on opponent lichen that isn't
  eliminable by a single dig action.

The action handling logic will also cancel conflicting actions (canceling conflicting
actions instead of attempting to resolve them):
- Cancel moves if they are to a stationary own unit, unit to be spawned, or into the
  destination of another moving own unit. This is done iteratively until no more
  collisions occur.
- Cancel transfers if they aren't going to a valid target (no unit or factory or unit or
  factory is at capacity)
- Cancel pickups if multiple units are picking up from the same factory and they'd cause
  the factory to go below 150 water or 0 power.

## Neural Architecture
I took a similar neural architecture to [FLG's
DoubleCone](https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406702), but
added an additional 4x-downsampling layer within the original 4x-downsampling layer to get
the receptive field to 64x64:

|                               |                  |
| ----------------------------- | ---------------- |
| levels                        | 3                |
| encoder residual blocks/level | [3, 2, 2]        |
| decoder residual blocks/level | [3, 2]           |
| stride per level              | [4, 4]           |
| deconvolution strides per     | [[2, 2], [2, 2]] |
| channels per level            | [128, 128, 128]  |
| trainable parameters          | 4,719,403        |
| value output size             | 14               |
| value output activation       | identity         |
| policy output per position    | 29               |

The policy output consists of 24 logits for unit actions, 4 logits for factory actions,
and 1 logit for factory placement. Each unit's action type and subactions are assumed
independent and identically distributed, as is the factory actions. The factory
placement logit is used to compute a probability of factory placement across all valid
factory spawn positions (all factory spawn positions are masked out if it's not the
agent's turn to place factories).

## Training Schedule
Similarly to the [2023 microRTS competition](../../microrts/technical-description.md), I
progressively trained the model on larger maps, starting with 16x16, then 32x32, and
finally 64x64. The best performing agent had the following training runs:

| Name                                                                                                                                                  | Map Size |
| :---------------------------------------------------------------------------------------------------------------------------------------------------- | -------: |
| [ppo-LuxAI_S2-v0-j1024env16-80m-lr30-opp-resources-S1-2023-11-16T23:18:33.978764](https://wandb.ai/sgoodfriend/rl-algo-impls-lux-nips1/runs/jk8u688d) |    16x16 |
| [ppo-LuxAI_S2-v0-j1024env32-80m-lr20-2building-S1-2023-11-18T09:16:46.921499](https://wandb.ai/sgoodfriend/rl-algo-impls-lux-nips1/runs/ewbq4e71)     |    32x32 |
| [ppo-LuxAI_S2-v0-j512env64-80m-lr5-ft32-2building-S1-2023-11-19T09:30:01.096368](https://wandb.ai/sgoodfriend/rl-algo-impls-lux-nips1/runs/idaxlrl0)  |    64x64 |

Each larger map training run was initialized with the weights from the best performing
checkpoint of the previous map size. The 16x16 map training run's weights were
initialized randomly.

I used my own implementation of PPO with the following hyperparameters:
| map size              | 16    | 32   | 64  |
| --------------------- | ----- | ---- | --- |
| batch size            | 32768 | "    | "   |
| mini-batch size       | 4096  | 1024 | 128 |
| autocast loss         | TRUE  | "    | "   |
| clip_range            | 0.1   | "    | "   |
| value clip range      | NONE  | "    | "   |
| GAE λ                 | 0.95  | "    | "   |
| γ (discount factor)   | 1     | "    | "   |
| gradient accumulation | TRUE  | "    | "   |
| epochs per rollout    | 2     | "    | "   |
| normalize advantage   | TRUE  | "    | "   |
| value loss halving    | TRUE  | "    | "   |

Each training run was for 80 million steps with the following schedule for learning rate
and entropy coefficient (cosine interpolation during transition phases):
| 16x16         | Start           | Transition ->1 | Phase 1           | Transition 1->2 | Phase 2           | Transition 2-> | End             |
| ------------- | --------------- | -------------- | ----------------- | --------------- | ----------------- | -------------- | --------------- |
| steps         |                 | 4M             |                   | 36M             | 32M               | 8M             |                 |
| entropy coef  | 0.01            |                | 0.01              |                 | 0.001             |                | 0.0001          |
| learning rate | 10<sup>-7</sup> |                | 3×10<sup>-4</sup> |                 | 5×10<sup>-5</sup> |                | 10<sup>-6</sup> |

| 32x32         | Start           | Transition ->1 | Phase 1           | Transition 1->2 | End             |
| ------------- | --------------- | -------------- | ----------------- | --------------- | --------------- |
| steps         |                 | 8M             | 24M               | 48M             |                 |
| entropy coef  | 0.001           |                | 0.001             |                 | 0.0001          |
| learning rate | 10<sup>-7</sup> |                | 2×10<sup>-4</sup> |                 | 10<sup>-6</sup> |

| 64x64         | Start           | Transition ->1 | Phase 1           | Transition 1->2 | End             |
| ------------- | --------------- | -------------- | ----------------- | --------------- | --------------- |
| steps         |                 | 8M             | 24M               | 48M             |                 |
| entropy coef  | 0.01            |                | 0.001             |                 | 0.00001         |
| learning rate | 10<sup>-7</sup> |                | 5×10<sup>-5</sup> |                 | 10<sup>-7</sup> |


Training was done on Lambda Cloud GPU instances each with 1 Nvidia A10. I also
used Nvidia A100 instances for the larger maps (not these specific training runs) where
I could double the mini-batch size. I used PyTorch's autocast to bfloat16 to reduce
memory usage and gradient accumulation to take optimizer steps on the full batch.

## Reward structure
RL solutions from the prior Lux Season 2 competition had to start training with shaped
rewards. Similar to my prior solution, I used generation and resource statistics to
generate the reward. However, instead of determining the scaling factors myself, I
normalized each statistic by dividing each statistic by its exponential moving variance
(window size 5 million steps). The environment would return all of these normalized
statistics and a WinLoss reward (+1 win, -1 loss, 0 otherwise), and the rollout would
compute an advantage for each statistic. The PPO implementation has element-wise scaling
factors for each advantage and reward for computing policy and value losses:
|                    | value coef | reward weights - 16 | reward weights - 32 | reward weights - 64 |
| ------------------ | :--------: | :-----------------: | :-----------------: | :-----------------: |
| WinLoss            |    0.2     |          1          |          1          |          1          |
| ice                |    0.1     |         0.1         |         0.1         |         0.1         |
| ore                |    0.1     |         0.1         |         0.2         |         0.2         |
| water              |    0.1     |         0.1         |         0.1         |         0.1         |
| metal              |    0.1     |         0.1         |         0.2         |         0.2         |
| power              |    0.1     |          0          |          0          |          0          |
| light bots         |    0.1     |         0.1         |         0.2         |         0.2         |
| heavy bots         |    0.1     |         0.1         |         0.2         |         0.2         |
| opponent bot kills |    0.1     |         0.1         |         0.2         |         0.2         |
| lichen             |    0.1     |          0          |          0          |          0          |
| factories          |    0.1     |         0.1         |         0.1         |         0.1         |
| # steps            |    0.1     |          0          |          0          |          0          |
| opponent lichen    |    0.1     |          0          |          0          |          0          |
| opponent factories |    0.1     |          0          |          0          |          0          |

The advantage of the above was that I could keep the same model and simply change the
weights in the value and reward coefficients to adjust the strategy. For example, the
training runs for 32x32 and 64x64 maps rewarded building robots more by increasing the
weighting of ore, metal, and robot generation.

## Supplementary Tables
Environment hyperparameters:
| map size                       | 16   | 32    | 64    |
| ------------------------------ | ---- | ----- | ----- |
| \# envs                        | 1024 | "     | 512   |
| rollout steps                  | 32   | "     | 64    |
| \# envs reset every rollout    | 128  | "     | 64    |
| allow adjacent factories       | TRUE | FALSE | FALSE |
| \# envs vs checkpoints         | 4×64 | "     | 4×32  |
| start bid                      | 0    | "     | "     |
| disable cargo pickup           | TRUE | "     | "     |
| enable light water pickup      | TRUE | "     | "     |
| disable unit-unit transfers    | TRUE | "     | "     |
| enable factory-digger transfer | TRUE | "     | "     |
| min water to lichen            | 1000 | "     | "     |
| factory max distance ice       | 0    | "     | "     |
| factory max distance ore       | 0    | "     | "     |
| union ice/ore spawn masks      | TRUE | "     | "     |
| initial factory water          | 150  | "     | "     |
| map min ice                    | 2    | "     | "     |
| map min ore                    | 2    | "     | "     |
| map min factories              | 1    | 2     | 4     |
| map max factories              | 2    | 3     | 10    |