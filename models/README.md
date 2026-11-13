## Included models

Two generations of deepCRE models are served, 36 files in total. Both are kept
on purpose: the tutorial and the published supplementary figures were produced
with the 2024 models, so they stay reproducible on the platform, while the
models of the current manuscript are what a user should choose for new work.

| set | files | naming | what it is |
|---|---|---|---|
| deepCRE 2026 — single species | 26 | `<Species>_<tissue>_v26J.h5` | 13 species x leaf/root, trained 2026-10-15 |
| deepCRE 2026 — multi-species  | 2  | `MSR_all_species_<tissue>_v26J.h5` | all 13 species, *Brassica oleracea* held out |
| Peleke et al., 2024           | 8  | `<Species>_<tissue>.h5` | *A. thaliana*, *S. lycopersicum*, *S. bicolor*, *Z. mays*, trained 2024-11-02/04 |

`manifest.tsv` in this folder is the authoritative list and drives the model
menu in the sidebar: one row per served file with its version, kind, species or
training set, tissue, validation fold or held-out species, menu label and
sha256. The app groups the menu by that table, so a species that a given set
does not contain cannot be selected.

**After any deployment the sha256 column must be re-checked against the files
actually served.** That hash is what lets a reader verify that the models behind
a published figure are the models on the platform; a manifest that no longer
matches the deployment certifies nothing.

The 16 round-2 models dated 2025-09-03, and the two *H. vulgare* files that were
never deployed, were removed: no published result rests on them and the 2026
models supersede them.
