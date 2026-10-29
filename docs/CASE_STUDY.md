# Case Study - Aligning a policy from noisy preferences

This walk-through follows the full pipeline, from a hidden reward to an aligned policy, and
shows where the method earns its result.

## The task
A hidden gold weight vector defines the true reward of every response to every prompt. The
learner never sees it. The goal is to make a policy prefer high-reward responses using only a
weakly aligned supervised dataset and a set of noisy pairwise comparisons.

## Stage one: a mediocre reference
A low-temperature teacher samples demonstrations that only weakly favor good responses. Behavior
cloning fits a reference policy to them by penalized maximum likelihood. The result is a
reference that is better than chance but far from optimal: it reaches an expected reward of
0.589 against an optimal of 1.144. This is the realistic starting point of post-training, a
supervised model that captures preferences imperfectly.

## Stage two: noisy preferences
A Bradley-Terry annotator produces 4000 pairwise comparisons. It prefers the higher-reward
response with probability `sigmoid(reward difference)`, so a sizeable fraction of comparisons
are flipped. No comparison reveals the reward directly; the signal is only in the aggregate.

## Stage three: Direct Preference Optimization
DPO aligns a copy of the reference by minimizing `-log sigmoid(margin)` over the comparisons,
where the margin is the beta-scaled difference of the chosen and rejected log-probability ratios
to the reference. The gradient is exact and the expected-feature terms cancel, so each update is
a nudge toward the chosen response and away from the rejected one, scaled by how wrong the
current margin is.

## The result
From 0.589, the aligned policy reaches 1.067 expected reward, closing 86 percent of the gap to
optimal, while staying within a mean KL of 0.782 of the reference. The decisive evidence is
indirect: the reward the learner never saw is reconstructed by the policy itself. The implicit
reward `beta * log (policy / reference)` correlates with the hidden true reward at r = 0.767,
and up to 0.888 at smaller beta. Alignment here is not memorization of labels; it is the
recovery of a latent reward from noisy comparisons.

## The lesson in the tradeoff
Pushing beta smaller recovers the reward more faithfully but moves the policy further from the
reference and risks fitting the label noise. The evaluation report sweeps beta so this frontier
is explicit, which is the honest way to present a method whose quality depends on a knob.
