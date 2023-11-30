# RL PPO Self-play Solution (Digs and Transfers Water, Survives Full Game)

Kaggle code submission: [https://www.kaggle.com/code/sgoodfriend/luxs2-rl-algo-impls-v0-0-12-rl-ppo-selfplay](https://www.kaggle.com/code/sgoodfriend/luxs2-rl-algo-impls-v0-0-12-rl-ppo-selfplay)

Training repo: [https://github.com/sgoodfriend/rl-algo-impls/releases/tag/v0.0.12](https://github.com/sgoodfriend/rl-algo-impls/releases/tag/v0.0.12) (the specific commit the model was trained on was [1c3f35f](https://github.com/sgoodfriend/rl-algo-impls/tree/1c3f35f47cfcdb542e10e1666c0a0968dc6f7779))

Weights & Biases report: [https://api.wandb.ai/links/sgoodfriend/v9g3qfbd](https://api.wandb.ai/links/sgoodfriend/v9g3qfbd)

After 20 million steps, the agent learned the following skills:

- Dig for ice and transfer the ice to base. The agent usually manages to survive the game if the opponent doesn’t intervene.
- Grow lichen on the map over the course of the game.

![Screenshot 2023-04-26 at 1.01.04 PM.png](lux_s2_writeup/Screenshot_2023-04-26_at_1.01.04_PM.png)

## RL Algorithm: PPO

The PPO implementation mostly reproduces [vwxyzjn/ppo-implementation-details/main/ppo_multidiscrete_mask.py](https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_multidiscrete_mask.py) and uses similar hyperparameters to the Gym-μRTS paper [1]. Lux uses a much bigger map (48x48 vs 16x16 in the Gym-μRTS paper), which required a lot more memory to train. In order to get it to fit I reduced the `batch_size` to 1152 (from 3072) for an Nvidia A10 GPU and 2304 for an Nvidia A100. `n_steps` was reduced to keep the 4 minibatches per epoch.

Initial bidding is a random distribution with mean 0 and standard deviation 5. Factory placement logic uses the logic from Abhishek Kumar’s [Building a Basic Rule Based Agent in Python](https://www.kaggle.com/code/scakcotf/building-a-basic-rule-based-agent-in-python) with a slight weighing towards building factories near opponents.

**********************************************************“Spike” learning rate decay:********************************************************** Instead of using linear decay for learning rate, I implemented a learning rate schedule that started 100 times lower, raises to peak at 10% of total timesteps, and drops to 1000 times lower at the end. I hoped the initially low learning rate would allow me to use a higher peak learning rate (`4e-4`, about 60% higher than Gym-μRTS), since the biggest errors and jumps occur at the beginning. The estimated KL-divergence hovered around 0.005-0.02 for much of the training, not catastrophic, but a higher than the standard recommended 0.01 threshold.

**Transitioning dense reward to sparse(-ish) reward over the course of training:** At the beginning of training, the agent gets rewards for digging and transferring raw resources to the factory and gets penalized for losing factories. At the end of training, the agent only gets rewarded/penalized for win/loss, lichen generation, and difference vs opponent in lichen generation. In between, the reward is a linear interpolation based on number of training steps.

**********************Training curve:********************** The chart below shows ice generation significantly climbing at just before 5 million steps. At 3.3M steps, the agent is averaging a dig action’s worth of ice. By the end, ice generation averages over 40K . Water generation isn’t shown because it closely mimics ice except 4-fold less accounting for the conversion ratio. Episode length lags behind ice generation a bit, but crosses the important 150 step threshold at 5.2M steps (each factory was given 150 water+metal [last factory got less for any used in the bidding]). The training mean score tapers down to ~5 by the end because the reward by the end is primarily win/loss (+10/-10).

![Screenshot 2023-04-28 at 9.01.06 AM.png](lux_s2_writeup/Screenshot_2023-04-28_at_9.01.06_AM.png)

## Autocurriculum

The agent trained with half the agents playing themselves while the other half played agents with weights from prior versions. The prior agents were selected from snapshots taken every 300,000 steps with a window size of 33 (~10M step history). Every 144,000 steps (6000 steps per environment) the weights are switched out randomly.

## Model: U-net-like

The Gym-μRTS paper uses an encoder-decoder and other competitors used ResNet/IMPALA networks; however, I went with U-net-like as it was similar to the paper’s encoder-decoder with the addition of skip connections at each resolution level. The model scaled down the map 4 times to a size of 3x3 at the bottom. The model had ~3.8M parameters.

****************Critic:**************** The value head was attached to the bottom of the U-net with an AvgPool2D to allow the map size to change while using the same model.

****************Output:**************** The actor head generates a MultiDiscrete action output for every grid position (what the paper refers to as Gridnet). Each grid position has the following discrete action outputs (30 bools):

- Factory action (4, includes do nothing)
- Unit action
    - Action type (6)
    - Move direction (5)
    - Transfer direction (5)
    - Transfer resource type (5)
    - Pickup resource type (5)

Transfer, pickup, and recharge action amounts are the maximum capable by the robot. The actions are enqueued repeating effectively indefinitely.

## Input

The LuxAI_S2 PettingZoo ParallelEnv is wrapped in gym Wrappers and an AsyncVectorEnv reimplementation.

****************Mapping the Lux observation to Box:**************** The Lux observation dictionaries are converted to a Box space of WxH and ~80 boolean and float outputs. Each grid position has information on the following:

- Board ice, ore, rubble, lichen;
- Factory ownership, tile occupancy, cargo, and water survival info;
- Unit ownership, cargo, and enqueued action;
- Turn, day index, and day cycle info.

************Invalid action mask:************ Actions that would be invalid to the LuxAI_S2 env are masked out by setting the logits of such actions to extremely large negative values, effectively zeroing the probability and gradients. Invalid actions masked out include:

- Not enough power to even enqueue actions
- Not enough power to move to a location (whether base power requirements or rubble) or illegal move into opponent factory
- No chance a transfer would lead to a valid destination (so if another friendly robot could be in that location, it’s not masked)
- Nothing possible to pickup
- Not enough power to dig or self-destruct

Generating the above action masks “appears” to be very expensive. Training steps per second dropped from ~120/second to starting at 20-30/s. The reason I say “appears” is that over the course of training, training steps/second eventually climbs to ~250/s. My guess is that early slowness is from the game resetting every ~15 steps (without masking the agent picks invalid actions doing nothing vs the masked agent doing actions that cause near instant defeat):

![Screenshot 2023-04-28 at 8.40.10 AM.png](lux_s2_writeup/Screenshot_2023-04-28_at_8.40.10_AM.png)

## Performance

The above chart on training steps/second probably explains why my attempts at improving performance through increased caching and asynchronous execution of environments did not help very much. Even using a modified version of gym’s AsyncVectorEnv to handle passing along action masks through shared memory ([LuxAsyncVectorEnv](https://github.com/sgoodfriend/rl-algo-impls/blob/1c3f35f47cfcdb542e10e1666c0a0968dc6f7779/rl_algo_impls/shared/vec_env/lux_async_vector_env.py)) helped little.

## Reproducibility

The submitted agent was the first model to exceed mean episode length above 150 steps (each factory got at most 150 water). This important milestone meant the model learned how to dig for ice and transfer the ice to the factory. This agent was also the best performing agent, outperforming followup agents during the competition and apparently outperforming an agent trained on the same commit:

![Screenshot 2023-04-30 at 12.38.43 AM.png](lux_s2_writeup/Screenshot_2023-04-30_at_12.38.43_AM.png)

The submitted agent is red. Blue is an agent trained at [4f3a500](https://github.com/sgoodfriend/rl-algo-impls/commit/4f3a500cf1e575c55b06ebc2409d6537493f9207) (immediately after the submission version [v0.0.12] and meant to use the same hyperparams as the submission agent). The green line is being trained on the same commit as the submission, but it did worse than both!

The submission agent isn’t a one-off. The blue line is trending towards 1000 step episode lengths (and most runs near the end are averaging over 990 steps [faded blue line]). With more steps and tweaking of the reward shape, training would likely catch up. However, this demonstrates how the same code and hyperparams can lead to drastically different training results.

## Transfer Learning from a Simpler Map

The same U-net-like model can run on different sized maps since the actor is entirely convolutions-based and the critic uses an AvgPool2D before its fully-connected layers. Therefore, I tried training the agent on a smaller map (32x32) with one factory each for 20M steps and used this trained model as a starting point for the same 20M step full-game training.

The simple-env trained agent was promising. After 20M “pre-training” steps, the agent was producing water reliably (though at an average of 10 per episode, which isn’t enough to survive).

Even more promising, if this same model was the starting point of the full-game training, the agent nearly immediately lasts over 150 steps. However, the agent took a while to get out of this local optima, taking ~15M steps before exceeding 300 steps in episodes. By 20M steps, episodes were going for ~600 steps, despite likely collecting enough ice to survive (15K ice → ~3750 water → 3.75 factory-games). This is about one-third of the ice of the submission agent.

![Screenshot 2023-04-28 at 1.26.14 PM.png](lux_s2_writeup/Screenshot_2023-04-28_at_1.26.14_PM.png)

## Next Steps

20M steps is not nearly enough for RL in a game of this complexity. With more time, I plan on training the agents for 100M+ steps, but with the following improvements:

- **Distinct phases for reward weights, `gamma`, and `gae_lambda`.** The thought is to have a Phase 1 where rewards are skewed towards mining and survival lasting 30-50% of the training depending on total steps. Phase 3 will be sparse win/loss + score difference rewards (with higher `gamma` [0.999] and `gae_lambda` [0.99] for end-game only rewards), with a transitionary Phase 2. This resembles the Lux Season 1 paper using training phases [2], but with the addition of a linear transition period.
- ******************************Switch from lichen generation to power generation.****************************** Lichen generation is problematic because lichen can be lost by not watering, which the agent could exploit. “Current lichen” is a better metric, but I’m going to switch to power generation, which combines factory survival, number of robots, and number of lichen tiles.
- **Asynchronous rollout generation instead of asynchronous rollout step generation.** LuxAsyncVectorEnv isn’t that efficient because if one of the 18 environments is resetting, then the entire step has to wait for map generation to complete. Even in the best case scenario of 1000-step games, a reset in one of the environments will happen every 1/55 steps. If instead each rollout of 192 `n_steps` (A10 GPU) runs independently, then the resets are handled more equally (at most once per environment for 1000-step games, almost one-quarter reset pauses). This effect should be significantly larger at the beginning of training.

The goal is for the agent to learn more advanced behaviors such as building robots, clearing rubble, direct competition with the opponent, and late-game strategies.

## References

[1] Huang, S., Ontañón, S., Bamford, C., & Grela, L. (2021). Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement Learning. arXiv preprint [arXiv:2105.13807](https://arxiv.org/abs/2105.13807).

[2] Chen, H., Tao, S., Chen, J., Shen, W., Li, X., Yu, C., Cheng, S., Zhu, X., & Li, X. (2023). Emergent collective intelligence from massive-agent cooperation and competition. arXiv preprint [arXiv:2301.01609](https://arxiv.org/abs/2301.01609).
