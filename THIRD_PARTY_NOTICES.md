# Attribution and licence boundaries

The original organiser source and documentation retain their MIT licence and
copyright notice in [LICENSE](LICENSE). [CITATION.cff](CITATION.cff) preserves
the organiser repository's citation metadata.

The learned components were trained using IDSSE, released with Deutsche Fussball
Liga (DFL) authorisation under **CC BY 4.0**. Credit Manuel Bassek, Robert Rein,
Hendrik Weber and Daniel Memmert (2025), *An integrated dataset of spatiotemporal
and event data in elite soccer*, *Scientific Data* 12:195.
[Publication](https://doi.org/10.1038/s41597-025-04505-y),
[dataset](https://doi.org/10.6084/m9.figshare.28196177).
The dataset's licence is distinct from the repository's source-code licence.
The current release contains trained parameters and preprocessing statistics,
not the external dataset download. Model provenance is recorded in
`models/continued-verifier/training_provenance.json`.

Supplied competition videos, tracking, tagging and homography are separately
licensed. This repository's MIT licence grants no rights to those inputs. Users
must obtain their own authorised bundle. The previously published organiser
public labels and baseline CSVs remain in earlier Git history; those historical
files were verified against the organiser's public repository. They are not
inputs to this model and are absent from the current release tree.

Python, PyTorch, NumPy, pandas, scikit-learn, Numba and the other pinned runtime
packages retain their own licences. Installing dependencies does not relicense
their contents under this repository's licence.
