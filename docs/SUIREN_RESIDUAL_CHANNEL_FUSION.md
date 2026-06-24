# Suiren Residual Channel Fusion

Branch: `codex/suiren-atom-residual-channel-fusion`

## Motivation

The previous channel experiment made Suiren atom channels independent from the
base atomic-number embedding and fused them only after all MPNN layers. That
changed the pretrained encoder input distribution too aggressively. This branch
keeps the main simple-add path for baseline comparability, but adds residual
channels inside the atom update path.

## Atom Inputs

The adapter still builds the baseline atom token as:

```text
atom_tokens = z_embedding + suiren_2d_projection + suiren_3d_projection
```

This `atom_tokens` value is used to construct the initial reaction token, so the
reaction-token input remains comparable to the main branch.

When Suiren atom features are available, the adapter also provides residual
channels:

```text
channel 0: z_embedding
channel 1: z_embedding + suiren_2d_projection
channel 2: z_embedding + suiren_3d_projection
```

Missing channels are represented by a channel mask.

## Encoder Update

For each MPNN layer, all channels pass through the same pair-conditioned message
layer with shared weights. After that layer, auxiliary channels produce residual
deltas relative to the base channel:

```text
delta_c = channel_c - base
base = base + gate_c(base, channel_c) * delta_c
channel_c = base + delta_c
```

The gate is initialized near zero, so the branch starts close to the pretrained
base encoder and learns Suiren residuals only when useful.

## Auxiliary Masked Path

The masked-edit auxiliary path does not have Suiren atom features, but it now
emits a base-only channel. With one channel, the residual-channel encoder reduces
to the original MPNN update while using the same encoder path as the property
forward.

## Suggested Ablations

Use the same Stage C finetuning protocol and seed-matched Stage A encoders:

- `base+2d`: only pass 2D atom caches.
- `base+3d`: only pass 3D atom caches.
- `base+2d+3d`: pass both 2D and 3D atom caches.

Compare against the main simple-add baselines for the same cache inputs. If the
residual-channel branch is working as intended, it should at minimum stay close
to simple-add when gates remain small, and improve when channel-specific deltas
help.
