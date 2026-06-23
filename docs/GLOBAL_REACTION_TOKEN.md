# Global Reaction Token

Branch: `codex/global-reaction-token`

## Motivation

The main branch constructs an initial `reaction_token`, keeps it fixed during
all atom/pair message-passing layers, and updates it only once at the end from
the final atom mean pool. In Suiren fusion, graph-level Suiren features are
added to this initial token, but they do not influence intermediate atom updates.

This branch treats the reaction token as a global virtual node that participates
in every encoder layer.

## Method

The encoder still consumes the same structured input:

```text
atom_tokens:     [B, N, H]
pair_tokens:     [B, N, N, H]
reaction_token:  [B, H]
```

Within each layer, the update order is:

```text
1. atom_h = pair_message_layer(atom_h, pair_h)
2. pooled = mean_pool(atom_h)
3. reaction_h = LayerNorm(reaction_h + reaction_update([reaction_h, pooled]))
4. atom_h = LayerNorm(atom_h + atom_global_update([atom_h, reaction_h]))
```

This adds two directions of communication at every depth:

- atom -> reaction: the global token absorbs the current atom state summary
- reaction -> atom: each atom receives the current global reaction context

The final `reaction_h` is the per-layer updated global token. The output shape
is unchanged.

## Relationship to Suiren Fusion

In the main branch, Suiren graph features enter as:

```text
reaction_token += projection(suiren_graph_features)
```

but this token only affects the final readout. With this branch, Suiren graph
information can influence every atom update through the global token path:

```text
suiren_graph -> reaction_h(layer k) -> atom_h(layer k)
```

This is closer to true graph-level conditioning than a final readout-only token.

## Checkpoint Compatibility

The branch adds `atom_global_update` and `atom_global_norm` modules to the
encoder. `load_pretrained_encoder` allows these new keys to be missing when
loading a Stage A encoder checkpoint, while still rejecting unrelated missing or
unexpected keys.

## Ablation Design

Use identical data splits, seeds, optimizer, batch size, early stopping, and
loss weights. Do not combine this branch with pair attention or attention
readout until each isolated experiment is understood.

| Experiment | Branch | Suiren cache | Geometry | Pretrained encoder | Purpose |
|---|---|---|---|---|---|
| B0 | `main` | off | 2D | off | Random encoder baseline |
| B1 | `main` | off | 2D | Stage A | Standard encoder transfer |
| G1 | `codex/global-reaction-token` | off | 2D | Stage A | Global token effect without Suiren |
| B2 | `main` | off | IRC-RP | Stage A | Standard 3D transfer |
| G2 | `codex/global-reaction-token` | off | IRC-RP | Stage A | Global token effect with 3D pair inputs |
| C0 | `main` | 2D+3D atom/graph | IRC-RP | Stage A | Current Suiren fusion baseline |
| G3 | `codex/global-reaction-token` | 2D+3D atom/graph | IRC-RP | Stage A | Whether Suiren graph conditioning benefits from per-layer global exchange |

Report validation loss, test MAE for `dE` and `dE_dagger`, early-stop epoch,
and training stability. The most important comparison is `C0` vs `G3`, because
that isolates whether graph-level Suiren descriptors should condition all atom
updates instead of only the final reaction readout.
