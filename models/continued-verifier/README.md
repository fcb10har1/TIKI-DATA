# Selected continued verifier

This directory is one pipeline with four required learned components:

- `historical.pt`: speed-masked 19-feature residual TCN.
- `segment_seed42.pt`: continued segment proposer, selected epoch 10.
- `segment_seed43.pt`: original second segment proposer.
- `verifier_seed42.joblib`: 128-tree depth 10 correction forest.

`configuration.json` freezes inference settings, asset/source hashes and the
reference CSV hash. `selection.json` is the portable active selection.
`training_provenance.json` documents labels, splits and exported checkpoint
provenance; `external_metrics.json` contains aggregate external results only.
Checkpoints contain model weights and required preprocessing metadata.
Optimizer state, absolute local paths and training caches are excluded.

Use the root README to regenerate the submission. Only load the bundled,
hash-verified joblib file; it is an executable serialization format.
The weights were trained on external IDSSE data; attribution is in TIKIDATA.md.
