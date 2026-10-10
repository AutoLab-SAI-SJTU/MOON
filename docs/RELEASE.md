# Release scope

Included: inference, simulator evaluation, adapter loading, fixed thresholds, metrics and CPU/model-free tests. Withheld: optimization loops, judge labeling, training-state mining, GRPO objectives/rewards, expert training rollouts and training configs. Training code is planned after acceptance.

An inference-only working tree does not remove training code from old commits. Public visibility exposes reachable history and branches. A clean source archive from this tree contains no `.git` directory or earlier training version.

Keep the development repository private. Publish the clean snapshot into a repository with clean history, or have the owner complete a deliberate history-isolation plan before changing visibility. Merging file deletions alone does not hide the original training implementation.

Model/data resources are separate. Follow their ModelScope permissions and licenses; do not commit credentials, model binaries, dataset archives, local outputs or private training logs.
