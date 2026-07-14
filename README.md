# RFM / Reaction-QM

Reaction-QM reaction encoder training code. The repository contains the
reusable Python package, training entry points, and stable interface
documentation.

## Training Entry Points

```bash
PYTHONPATH=. python -m rfm.cli.train_masked_edit ...
PYTHONPATH=. python -m rfm.cli.train_reaction_property ...
PYTHONPATH=. python -m rfm.cli.train_suiren_fusion_property ...
```

## Repository Layout

```text
rfm/        Python package for datasets, token-space features, models, losses, metrics, and optimizers.
docs/       Stable schema and code-structure documentation.
pyproject.toml
```

## Interface Docs

- `docs/CODE_STRUCTURE.md`
- `docs/MRTO_FULL_ARCHITECTURE.md`
- `docs/HDF5_PROCESSED_SCHEMA.md`
- `docs/ATOM_LEVEL_SUIREN_FEATURE_SCHEMA.md`
