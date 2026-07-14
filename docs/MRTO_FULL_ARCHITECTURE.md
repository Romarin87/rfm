# MRTO Full Architecture

`mrto_full` is the complete Mapped Reaction Transition Operator implementation.
It models an atom-mapped reaction as a transition rather than as post-encoder
fusion of two molecular embeddings.

## Input Contract

The active 3D path uses only official IRC R/P endpoint geometries. TS geometry is
never an input. R and P coordinates live in independent frames and are processed
by one weight-shared Suiren `EST_Eqv2`/EquiformerV2 endpoint adapter.

```text
2D raw: Z, BO_R, BO_P
3D raw: coordinates_R_irc, coordinates_P_irc
Stage C prior: frozen Suiren R/P atom and graph features
```

Stage A receives masked R-side state and `coordinates_R_irc`. Stage B/C receive
both endpoints for the property view and the same R-only view for the Stage A
loss. The property and masked-edit adapters share one endpoint Equiformer.

## Operator

```text
shared endpoint topology and EquiformerV2 geometry encoders
-> exact R/P even and odd atom/pair fields
-> parity-preserving low-rank triangle pair update
-> pair-biased atom attention
-> competitive sparse reaction-event routing
-> event-to-pair and event-to-atom feedback
-> even/odd reaction fields
```

The event bottleneck predicts an unordered edit set through event presence,
pair location, and signed bond-order change. Exact slot-permutation matching is
used for the small configured slot count.

Stage A heads receive `reaction_h` as global context in addition to atom/pair
states. This makes the reaction-level readout part of the actual Stage A loss
graph instead of leaving it as an unsupervised output-only projection.

Frozen Suiren features are input priors. Atom priors condition atom fields and
graph priors condition event fields inside every operator block with parity-aware
AdaLN/FiLM gates. They are not concatenated after the encoder and do not change
the output interface.

## Output Contract

All 2D, IRC-3D, and Suiren configurations preserve:

```text
atom_h: [B, N, H]
pair_h: [B, N, N, H]
reaction_h: [B, H]
```

MRTO-full additionally exposes typed even/odd fields and event assignments for
task heads and analysis. The Stage B/C energy head predicts odd `dE` and even
symmetric barrier `B_sym`, then reconstructs
`dE_dagger = B_sym + 0.5 * dE`. This enforces the exact reverse-reaction energy
relation by construction.

## Stage A Objective

```text
L_A = L_DeltaBO + 0.5 L_changed + 0.5 L_edit + 0.2 L_core
      + lambda_set L_edit_set
```

The first full run uses four event slots, top-16 candidate pairs per slot, and
`lambda_set = 0.1`. Existing event coverage/diversity surrogates remain disabled.

## Runtime Dependency

The Suiren source tree must be importable so `suiren_models.model.EST_eqv2` is
available. For the project workspace:

```bash
export PYTHONPATH=${RFM_CODE}:${BASE}/Suiren-Foundation-Model/src
```

The environment also requires PyTorch Geometric, `torch-cluster`, and `e3nn`.
