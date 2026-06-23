# Attention Reaction Readout

Branch: `codex/attention-reaction-readout`

## Motivation

The main branch feeds the property head with:

```text
concat(reaction_h, mean_pool(atom_h))
```

Uniform mean pooling can over-smooth atom information, especially when only a
small number of atoms around the reaction center dominate the target property.
This branch keeps the encoder unchanged and replaces the final atom mean pool
with a masked attention readout conditioned on the learned `reaction_h`.

## Method

After the encoder produces `atom_h` and `reaction_h`, a small attention module
uses `reaction_h` as the query and atom states as keys/values:

```text
q = W_q reaction_h
k_i = W_k atom_h_i
v_i = W_v atom_h_i
alpha_i = softmax_i(q dot k_i / sqrt(H)) over valid atoms
atom_pool = sum_i alpha_i v_i
y = property_head([reaction_h, atom_pool])
```

The module is applied to both regular reaction property regression and Suiren
fusion property regression. Masked edit auxiliary heads are unchanged.

## Scope

This experiment changes only the readout immediately before the property head.
It does not change:

- atom token construction
- pair token construction
- Suiren fusion method
- encoder message passing
- loss weights

The property head input dimension remains `[B, 2H]`, so checkpoint layout stays
close to the main branch except for the added readout parameters.

## Expected Effect

The readout can focus on atoms that are most relevant to the reaction-level
target instead of averaging all atoms uniformly. This should help cases where
large molecules contain many spectator atoms.

## Ablation Design

Use the same train/valid/test splits, seeds, optimizer, batch size, and early
stopping settings.

| Experiment | Branch | Suiren cache | Geometry | Pretrained encoder | Purpose |
|---|---|---|---|---|---|
| B0 | `main` | off | 2D | Stage A | Mean-pool Stage B baseline |
| R1 | `codex/attention-reaction-readout` | off | 2D | Stage A | Attention readout without Suiren |
| B1 | `main` | off | IRC-RP | Stage A | Mean-pool 3D baseline |
| R2 | `codex/attention-reaction-readout` | off | IRC-RP | Stage A | Attention readout with 3D pair inputs |
| C0 | `main` | 2D+3D atom/graph | IRC-RP | Stage A | Current Suiren fusion baseline |
| R3 | `codex/attention-reaction-readout` | 2D+3D atom/graph | IRC-RP | Stage A | Attention readout under Suiren fusion |

Report validation loss, test MAE for both targets, early-stop epoch, and whether
attention readout changes the largest failure cases. If possible, save atom
attention weights for a small fixed validation subset for qualitative review.
