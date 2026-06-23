# Pair Attention Aggregation

Branch: `codex/gat-pair-attention`

## Motivation

The main branch uses an edge-conditioned MPNN layer where every valid neighbor
contributes equally after the pair-conditioned message MLP:

```text
m_ij = MLP([h_i, h_j, pair_ij])
h_i' = Update(h_i, mean_j m_ij)
```

This is stable, but it assumes all valid pairs are equally useful for updating
an atom. Reaction property prediction is often dominated by a small subset of
pairs around the reaction center, changed bonds, or important long-range 3D
contacts. This branch keeps the same message inputs but replaces uniform mean
aggregation with a GAT-like attention readout over neighbors.

## Method

For each ordered atom pair `(i, j)`, the layer computes the usual
pair-conditioned message and an attention logit from the same representation:

```text
r_ij = [h_i, h_j, pair_ij]
m_ij = message_mlp(r_ij)
a_ij = attention_mlp(r_ij)
alpha_ij = softmax_j(a_ij over valid pairs)
agg_i = sum_j alpha_ij m_ij
h_i' = LayerNorm(h_i + update_mlp([h_i, agg_i]))
```

Invalid pairs and self pairs remain masked out through `pair_valid_mask`.
The encoder input and output contracts are unchanged:

```text
atom_tokens: [B, N, H]
pair_tokens: [B, N, N, H]
reaction_h:  [B, H]
```

The branch intentionally does not update `pair_h`; pair tokens stay as fixed
edge features so this experiment isolates the aggregation change.

## Expected Effect

The model can learn to give larger weights to chemically important neighbors
instead of averaging all pair messages uniformly. This should help when noisy or
uninformative unchanged pairs dominate the dense `[N, N]` pair set.

## Ablation Design

Run the same seeds, splits, cache inputs, optimizer, batch size, and early-stop
settings for all rows below.

| Experiment | Branch | Suiren cache | Geometry | Pretrained encoder | Purpose |
|---|---|---|---|---|---|
| B0 | `main` | off | 2D | off | Pure baseline without pretraining |
| B1 | `main` | off | 2D | Stage A | Baseline transfer benefit |
| A1 | `codex/gat-pair-attention` | off | 2D | Stage A | Effect of attention aggregation alone |
| C0 | `main` | 2D+3D atom/graph | IRC-RP | Stage A | Current Suiren fusion baseline |
| A2 | `codex/gat-pair-attention` | 2D+3D atom/graph | IRC-RP | Stage A | Pair attention under Suiren fusion |

Primary metrics should be validation loss selection, test MAE for `dE` and
`dE_dagger`, and failure-case distribution. Also record whether attention
aggregation changes training stability or early stopping epoch.
