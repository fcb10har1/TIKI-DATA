# Neural component: technical-document insert

We implemented a small offline residual Temporal Convolutional Network in
PyTorch. It accepts P1's ordered, match-relative numerical features and their
explicit observation flags, and predicts framewise probabilities in H,A,D
order. The default model has 32 hidden channels and five residual blocks with
two kernel-3 convolutions per block, dilated by 1, 2, 4, 8 and 16. Its 125-frame
receptive field uses approximately five seconds of past/future context at 25 Hz.
X denotes an unknown training target or decoder abstention, not a neural class.

Whole external matches are assigned to train, validation and final test before
window construction. Feature means and standard deviations use only observed
training frames within halves. Unobserved entries become zero after normalization
and are distinguished by appended flags. Independent halves have separate context;
window halos provide the entire receptive field, while cross-entropy ignores
padding and unknown labels. Inference retains each frame exactly once, including
short windows and remainder frames. The checkpoint includes all preprocessing
metadata. Targets/status fields and tactical tags are excluded from inputs;
collective features must not interpret tracker slots as player identities.

Training uses AdamW, clipping and early stopping. Validation selection uses mean
per-match official spellF1, then macroF1. The final external test is reserved for
P4 after freezing the full rules/model/decoder pipeline. The supplied match and
its public five-minute labels are not training or tuning data. P3 owns tactical
tag fusion, control/stoppage rules and Viterbi decoding.

IDSSE is a candidate external source with seven matches, accessible under CC BY
4.0 with credit to DFL and Bassek et al. (2025), DOI
[10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
Its home/away possession and active/inactive status codes have a target-only
H/A/D mapping. Its published control description does not establish the
competition's exact treatment of loose balls, deflections or control transitions;
scores on those annotations would measure transfer from a related definition.

A separate configurable augmentation hides feature observations and contiguous
ball-dependent feature stretches. This provides an execution path for robustness
experiments; it is not a demonstrated improvement. Raw-tracking masking followed
by feature recomputation remains a P1/P2 follow-up to avoid residual information
in derivatives and interpolation.

At delivery, only synthetic execution, alignment, training and checkpoint-reload
checks have run. No real football model has been trained and no real accuracy
or robustness gain is reported. Real experiments require P1's shared feature
extractor, aligned external targets, reviewed feature lineage, and agreed whole-
match splits. Reproduction commands and limitations are in
[P1-to-P2 V1 Handoff.md](P1-to-P2%20V1%20Handoff.md).
