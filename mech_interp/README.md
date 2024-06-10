# Toward a board presence linear probe on microRTS agent
Scott Goodfriend (goodfriend.scott@gmail.com)

_A project for the AI Safety Fundamentals Alignment course._

This is a first pass at creating a basic linear probe on an agent trained to play a
testbed real-bed strategy game microRTS. The goal of this project is to create a linear
probe that can predict the occupancy of the entire map from the residual stream of a
single unit entity. Despite trying variously trained agents, we did not find the linear
probe outperformed a baseline based only on grid position statistics. We cannot rule out
bugs, but these findings show that more nuanced probes (such as focusing on immediate
neighborhood versus entire board) are likely necessary to find measurable effects.

## Importance to AI Safety
Mechanistic interpretability of reinforcement learning (RL) trained agents is still in
its nascent stages compared to work done with large language models (LLMs). As we
develop AI that act as agents in the virtual and physical world, RL is a natural way to
train such agents. However, we don’t know if we can train agents using RL that won’t
learn to pursue long-term (possibly instrumental) goals. We already have concrete cases
of goal misgeneralization when using RL ("Goal Misgeneralization in Deep Reinforcement
Learning", Langosco et. al. 2022). Currently, we have little data on what’s going on
with these RL-trained models. Mechanistic interpretability case studies on capable
RL-trained models can go a long way towards understanding (and eventually trusting) agents
trained with RL.

I believe using a transformer neural network trained on microRTS is an excellent case study to get insight into what and how RL agents are learning:

1) A lot of recent interpretability advancements have been on transformer-based LLMs, which can be tried on this transformer model.
2) Representing each unit and resource as its own “token” is natural for a transformer and interesting circuits should form to represent relationships between units.
3) microRTS is easily customizable and editable in-game. This will help us control training diversity and to test for misgeneralization.
   
I’m not aware of any investigations on interpreting a centralized multi-agent model (a
single model controls the actions of many components). This could be an effective
strategy to control safety-critical processes that require high coordination (e.g.,
traffic, factory or warehouse operation). microRTS is an excellent case study for such
agents.

## Related Work
This project uses the work from 3 projects: 

1) [rl-algo-impls](https://github.com/sgoodfriend/rl-algo-impls): This reinforcement
   learning library was used to win the [2023 IEEE Conference on Games microRTS
   competition](https://arxiv.org/abs/2402.08112). This library is used to train the models and probes.
2) [entity-neural-network](https://github.com/entity-neural-network): This project
   represents entities as "tokens" in a transformer architecture. We use [their
   hyperparams for
   Gym-μRTS](https://github.com/entity-neural-network/enn-zoo?tab=readme-ov-file#gym-%C2%B5rts)
   for the initial reinforcement learning (RL) trained agent.
3) [ARENA_3.0 OthelloGPT Training a Probe
   implementation](https://arena3-chapter1-transformer-interp.streamlit.app/[1.6]_OthelloGPT): 
   While I started with RL agents, I switched to behavior cloning a random move bot because
   the [Emergent World Representations paper](https://arxiv.org/abs/2210.13382) found
   that training their GPT-2 transformer on a random move dataset instead of a
   championship move dataset generated a model whose probes could elicit board state
   better.

## Implementation
This project involved two major phases:

1) Implement linear probe training that can take a trained agent, attach probes to the
   residual stream of this agent, and train the linear probe as the agent played the
   game.
2) Train different agents to improve linear probe performance compared against a baseline
   estimate of board occupancy independent of the model.

### Linear Probe Trainer
This is primarily in [linear_probe_train.py](linear_probe_train.py) and
[occupancy_linear_probe_trainer.py](occupancy_linear_probe_trainer.py).
linear_probe_train.py loads the (1) trained agent model from Weights & Biases and (2) the
vectorized environment into an `OccupancyLinearProbeTrainer`. The
`OccupancyLinearProbeTrainer` (1) initializes the linear and bias parameters for the
linear probe, (2) hooks into the residual output of a transformer block
(`residual_layer_idx` layer index hyperparameter), (3) predicts occupancy of every grid point using the residual output as
input to the linear probe, (4) optimizes the probe parameters using the actual occupancy
as the target. For a given timestep, the linear probe makes N predictions where N is the
number of entities on the map. Therefore, the linear probe is determining if every
entity gains a full-board representation through the course of the transformer computing
the output.

#### Baseline Probe
`OccupancyLinearProbeTrainer` can also be trained using only the bias parameters
(zeroing out the weights of the linear parameters) through the `detach` hyperparameter.
This "baseline probe" represents board occupancy statistics (how often a board position
is occupied over the course of playthroughs) independent of the model's input and
transformer computation. Such a baseline is likely to be highly accurate because certain
board positions are likely to be occupied (e.g., the bases at the top-left and
bottom-right), unoccupied (e.g., the corners not occupied by the bases), and moderately
occupied (e.g., immediately around the bases and mildly the middle of the board).

The linear probe should be more accurate than the baseline probe, but this is dependent
on how the model is trained and understanding the challenges of this project.

### Challenges
While Othello-GPT was inspiration for this project, microRTS has several significant
challenges:
1) The input and output of the entity-based transformer is significantly different from
   a token-based sequence. Othello-GPT mapped naturally to a token sequence by
   representing each token as a board position. In contrast, microRTS's input sequence
   is the units and resources of the current timestep. While Othello-GPT's output is
   simply the last token, microRTS's output is on every entity that the player owns (on
   every turn, a microRTS player can move several units at once).
2) Invalid action masking is an additional output filter not used by Othello-GPT. RL
   microRTS [relies on filtering out illegal and no-op moves to reduce the exploration
   space](https://arxiv.org/abs/2006.14171). However, illegal action masking allows the
   model to behave reasonably without needing to be aware of illegal moves such as
   moving into occupied spaces.
3) A unit entity likely doesn't need to know full-board occupancy to predict it's next
   move. This means that putting a linear probe on the residual stream of a single unit
   entity likely wont' find full-board occupancy data encoded. In contrast, in
   Othello-GPT each move can have long-distance effects and legality can be based on
   far away pieces; therefore, encoding entire board state is more important for
   Othello-GPT.

### Training Agent Models
We trained 4 categories of agents:

1) PPO agent (PPO): Reinforcement Learning agent trained to defeat the 2020 microRTS competition winner
   CoacAI using Proximal Policy Optimization (PPO).
2) Behavior cloned (BC): Behavior cloned agent trained to imitate the behavior of the
   randomBiasedAI, a microRTS baseline agent that randomly picks actions but biases
   towards attacking an in-range enemy (neighbor for all non-ranged units).
3) Behavior cloned, no invalid action mask (BC, no mask): Similar to BC but
   the RL invalid action mask is replaced with a mask that sets all actions valid for
   own units ready to take commands (not currently doing an action) while otherwise keeping everything else invalid.
4) Behavior cloned, no invalid action mask, large (BC, no mask, large): Similar to "BC,
   no mask" but the model has twice as many transformer blocks (4 vs 2), twice the
   number of attention heads per layer (4 vs 2), and twice the residual dimension (64 vs 32).
   The intention is that this gives the model capacity to represent board position in
   the residual stream and the ability to compute such information with the additional
   layers and heads.

Many variations of the above 4 agents were trained, but these 4 models capture important
milestones.

## Results
Overall, the linear probe performed similarly to the baseline probe, sometimes
performing slightly worse.

### PPO
![The linear (yellow) and baseline (cyan) probes converge on nearly the same accuracy and
loss.](imgs/probe-RL.png)

* PPO trained agent: [ppo-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-05-07T20:34:29.247010
](https://wandb.ai/sgoodfriend/mech-interp-rl-algo-impls/runs/eh5nxxe2)
* Linear probe (yellow): [probe-ppo-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-05-22T21:06:03.901118
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/6nesdrrs?nw=nwusersgoodfriend)
* Baseline probe (cyan): [probe-detached-ppo-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-05-22T21:33:33.104123
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/ing5xwzz?nw=nwusersgoodfriend)

### BC
![The linear (brown) probe is slightly over 1% less accurate than the baseline (red)
probe.](imgs/probe-BC.png)

* BC trained agent: [acbc-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-06-08T06:11:30.418059
](https://wandb.ai/sgoodfriend/mech-interp-rl-algo-impls/runs/lfo9ll4m?nw=nwusersgoodfriend)
* Linear probe (brown): [probe-acbc-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-06-08T13:10:08.729669
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/xj2b2kv4?nw=nwusersgoodfriend)
* Baseline probe (red): [probe-detached-acbc-Microrts-b2w10-grid2entity-a10-multi-rews-max-critic-neck-S1-2024-06-08T13:19:03.235833
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/pldrxtwr?nw=nwusersgoodfriend)

### BC, no mask
![The linear (purple) and baseline (yellow) probes convene on nearly the same accuracy
and loss.](imgs/probe-noMask.png)

* "BC, no mask" trained agent: [acbc-Microrts-b2w10-grid2entity-ignore-mask-S1-2024-06-09T03:34:22.495925
](https://wandb.ai/sgoodfriend/mech-interp-rl-algo-impls/runs/ja5u6r9y?nw=nwusersgoodfriend)
* Linear probe (purple): [probe-acbc-Microrts-b2w10-grid2entity-ignore-mask-S1-2024-06-08T23:33:41.300964
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/kwpkdsji?nw=nwusersgoodfriend)
* Baseline probe (yellow): [probe-detached-acbc-Microrts-b2w10-grid2entity-ignore-mask-S1-2024-06-08T23:41:59.164532
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/y3gv5r5e?nw=nwusersgoodfriend)

### BC, no mask, large
"BC, no mask, large" probes only predict on entities that can take actions instead of
all entities (as done earlier). The
idea is that action taking entities are the only entities whose output is being trained
during behavior cloning. The results still show similar performance between baseline and
linear probes, so the earlier probes were not retrained.

![The linear probes (orange and green) and baseline (pink) probes convene on nearly the
same accuracy and loss](imgs/probe-noMaskLarge.png)

* "BC, no mask, large" trained agent: [acbc-Microrts-b2w10-grid2entity-ignore-mask-4layers-S1-2024-06-09T07:43:20.497034
](https://wandb.ai/sgoodfriend/mech-interp-rl-algo-impls/runs/1synx02r?nw=nwusersgoodfriend)
* Layer 1 linear probe (orange): [probe-acbc-Microrts-b2w10-grid2entity-ignore-mask-4layers-S1-2024-06-09T16:23:17.140649
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/cjepf215?nw=nwusersgoodfriend)
* Layer 2 linear probe (green): [probe-acbc-Microrts-b2w10-grid2entity-ignore-mask-4layers-S1-2024-06-09T16:04:41.294346
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/onpwwdoh?nw=nwusersgoodfriend)
* Baseline probe (pink): [probe-detached-acbc-Microrts-b2w10-grid2entity-ignore-mask-4layers-S1-2024-06-09T16:18:03.351128
](https://wandb.ai/sgoodfriend/rl-algo-impls-interp/runs/rj4mjikz?nw=nwusersgoodfriend)

_Note: Layer indexes are 0-indexed, so Layer 1 in a 4 layer network is after the second
transformer block._

## Conclusion
We did not train a linear probe that outperformed the statistical baseline. We cannot
rule out a bug, especially since we'd expect negligible gains from the additional data
from the linear parameters (at least it should know it's own position is occupied).
However, it is somewhat likely that each entity doesn't generate a full-board
representation to compute its next move. This is especially true for the random move
behavior cloning since those moves are computed independently of board state outside of
the immediate local neighborhood.

As discussed in the [Challenges section](#challenges), Othello-GPT's random valid moves
are dependent on long-ranged board positions, therefore such a model would need to
build an internal representation that accounts for the entire board.

## Possible Next Steps
We will likely not continue work on this project, but interested parties should consider
these possible next steps if they want to continue this or similar investigations:

1) Investigate if accuracy comes from the probe always predicting "not occupied" given total map
   occupancy likely lies around 10%. If this is the case, weigh the occupied group
   higher.
2) Linear probe occupancy on neighboring spaces around entities. The random move bot
   being behavior cloned only considers if neighboring spaces are occupied for targeting
   actions. A behavior cloned model likely learns only to consider neighboring spaces.
3) Attempt to recreate the action mask with the linear probe for each entity that is
   able to perform an action this turn. Only own units that are not already performing
   an action can take actions. The "BC, no mask" agents are trained without an invalid
   action mask, but a random bot will avoid invalid actions so an implicit invalid
   action mask could be learned from training.