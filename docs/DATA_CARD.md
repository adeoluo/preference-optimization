# Data Card - Preference Optimization

## Sources and provenance
All data is **generated** from a fixed alignment task defined in
`src/preference_optimization/domain.py`. A hidden gold weight vector defines the true reward of
every (prompt, response) pair as a normalized dot product with that pair's feature vector. Two
datasets are derived from this task and this task alone:

- **Demonstrations** for behavior cloning, sampled from a low-temperature teacher that favors
  higher-reward responses. The low temperature makes the teacher only weakly aligned, which is
  what produces a mediocre, improvable reference.
- **Preferences** for DPO, sampled from a Bradley-Terry annotator that prefers the
  higher-reward response in a pair with probability `sigmoid(reward difference)`, so roughly a
  third of comparisons are noisy flips.

## Licensing
The task, the feature vectors, and both generators are original to this project under the
repository's MIT license. No external corpus is used.

## Composition
A prompt is an integer index; a response is an integer index into that prompt's candidate set.
The default task has 6 prompts, 5 responses each, and 12-dimensional feature vectors. Training
uses 600 demonstrations and 4000 preferences; evaluation uses 800 held-out preferences drawn
from a disjoint seed.

## Preprocessing and filtering
None is required: the generators emit clean, fully specified records. There is no personally
identifiable information and no free text.

## Contamination checks
Training preferences, held-out preferences, and demonstrations use distinct seeds, so the
evaluation comparisons are independent of the training comparisons. Generation is deterministic,
so any overlap would be visible and reproducible.

## Known biases and gaps
The annotator noise follows the Bradley-Terry model exactly, which is more consistent than real
human labeling. The reward is linear in the features by construction, which a log-linear policy
can represent; natural tasks are not linear in any simple feature space.

## Ethical considerations
The data is synthetic and contains no personal or sensitive content. The main risk is reading
too much into alignment results obtained on a controlled bandit, which the model card and
README state explicitly.
