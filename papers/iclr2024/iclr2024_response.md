## General Response
We sincerely appreciate the feedback and insights provided by everyone. They are helpful in
identifying areas for improvement both in this paper and in ongoing work.

We've addressed reviewer-specific comments directly on the reviews. Additionally, we
want to address the following generally:

**Novelty to DRL field:** We agree with the interpretation that AnonymousAI uses many
best-practices across DRL. We believe the novelty of this paper is specifically in
microRTS and environments where many units have to be individually controlled. This is
an area where hand-scripted AI agents have been dominant, as evidenced by prior winners
of microRTS competitions and the latest Lux Season 2 competitions on Kaggle. This paper
represents a milestone for DRL in this domain on single GPU hardware, which is more
common in an academic setting.

Minor comments:
- We forgot to add the Behavior Cloning schedule for 16x16 maps. This has been added to the
  Behavior Cloning Supplementary Section.

## AVMG Responses
1. **Advanced self-play techniques:** In this paper we trained our agent against a
combination of scripted opponents, the current agent, and past versions of the agent.
Past versions of the agent were taken from periodic checkpoints during training, and the
selected version was chosen randomly. As suggested, using fictitious self-play
similar to TiZero or AlphaZero could be a promising direction for future work,
especially for larger maps where multiple strategies are likely viable. A paragraph has
been added to the Discussions sections.
1. **Novel behaviors from behavior cloning:** We see using the term "novel" was confusing.
We replaced most instances of "novel" with "demonstrated", "competitive", and
"effective".
1. **Training hardware details:** Sorry, this was an oversight. This has been added to
Training Durations Supplementary Section.

## nGmd Responses
** SCC and related works discussion:** Thanks for pointing us to the SCC paper. It's
definitely impressive to accomplish similar results to AlphaStar with a smaller model
and order of magnitude less compute. We knew of TStarBot-X as another successful
StarCraft II agent trained with orders of magnitude less computation scale. We didn't go
into these papers because these still required clusters of GPUs to train (SCC doesn't
give specifics but order of magnitude less is tens of TPUs still). We thought microRTS and the
Kaggle Lux competitions were more relevant given similar scale.

1. MicroRTS-Py had a bug where the second player would see resources (which should be
  unowned) as owned by the first player. This was fixed for our agent. We moved that
  mention up into the MicroRTS-Py section to make it more clear.
1. Behavior cloning treats a reinforcement learning problem as a supervised learning
   problem by making the ground truth the actions taken by the agent being cloned.
2. AnonymizedAI trained 4 models fine-tuned to specific maps. 3 "general" models were
   also trained for 3 different map size ranges (16x16 and smaller, up to 32x32, and up
   to 64x64). Table 5 in the Competition Details Supplementary Section shows the 7
   models' usage. The fine-tuned models significantly outperformed the general models on
   their specific maps.
3.  squnet is the name of our fewer parameter model. It is short for "Squeeze U-Net"
    because it uses squeeze excitation like DoubleCone while being shaped like a U-Net. It is not
    a U-Net.
4. **Applicability to other strategy games:** We focused on
microRTS because of its unique challenges and the existence of a benchmark without a
strong DRL agent. We have used the same techniques to train
agents for the Lux Season 2 competitions, which is a similar turn-based game requiring
the coordination of many units individually. We have not tried other games, but we
believe that many of these techniques would be applicable to other games. For example,
creating a curriculum of simpler scenarios and gradually increasing difficulty or using
a demonstration dataset to create a starting point for training are generally
applicable.

## 7eJX Responses
**Design decisions and ablation studies:** We agree that the paper could be improved by
a thorough analysis of design decisions. Unfortunately, we were time-constrained by the
competition deadline and were experimenting with many different ideas going up to the
deadline. We ran training runs with changes and kept the changes that made 
improvements or we believed would be useful down the line.

One example of this is the scaling the policy loss by the number of units that could
take an action, which was critical for our behavior cloning. We have added our reasoning on
why this is important; however, we believe this could be a candidate for an ablation
study to determine if any learning rate works without this scaling.

Other candidates for ablation studies include:
- transfer learning smaller map agents to larger maps
- varying reward weights, value loss coefficients, and entropy coefficients over time
- cost-based rewards

## Pjfo Responses
**Generalization to different maps:** In Table 4, we show that BC-PPO-Agent performs as
well as AnonymizedAI on the 3 maps AnonymizedAI needed fine-tuned models while only
using map-size specific models. We think this is a promising direction to
generalization, and we added a paragraph at the end of the Behavior Cloning Results
Section stating this.

**Generalization to larger maps:** We are working on generalizing to larger maps. We
extended the Training on Larger Maps Discussion Section to include our current ideas.

**What to action mask?:** We used MicroRTS-Py as the starting point and masked out a
couple more cases where the action would cause nothing to happen. Implementing action
masking is a major contribution of human knowledge to the learner but has been
well-established as critical for learning in microRTS.
