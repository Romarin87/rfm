# RFM Code Structure

This file describes the repository code layout. Training entry points live in
the `rfm/` package.

## 1. Package Layout

```text
rfm/
  data/
    hdf5.py                # packed HDF5 与 feature cache I/O
    reaction_samples.py    # HDF5 sample 读取、raw pair_input 与 target 组装
  features/
    token_space.py         # 派生特征、ReactionInputFeaturizer、adapter、RFMEncoderInput
    mrto_v1.py             # exact R/P even-odd fields and parity primitives
    mrto_full.py           # EquiformerV2 endpoints, sparse events, full MRTO operator
  models/
    task_models.py         # masked edit / reaction property regression 模型封装
  optim/
    builder.py             # AdamW / Muon optimizer builder
    muon.py                # Muon optimizer implementation
  tasks/
    metrics.py             # regression / binary metrics
    masked_edit.py         # masked edit dataset、loss、eval、summary
    reaction_property.py   # reaction property regression dataset、loss、eval、summary
    suiren_fusion_property.py # frozen Suiren cache + reaction property dataset
  cli/
    train_masked_edit.py        # masked edit pretraining 正式 CLI
    train_reaction_property.py  # reaction property regression 正式 CLI
    train_suiren_fusion_property.py # Suiren fusion property regression 正式 CLI
```

## 2. Official Entrypoints

推荐入口：

```bash
PYTHONPATH=. python -m rfm.cli.train_masked_edit ...
PYTHONPATH=. python -m rfm.cli.train_reaction_property ...
PYTHONPATH=. python -m rfm.cli.train_suiren_fusion_property ...
```

在工作目录中可以直接使用：

```bash
PYTHONPATH=${BASE} python -m rfm.cli.train_masked_edit ...
PYTHONPATH=${BASE} python -m rfm.cli.train_reaction_property ...
PYTHONPATH=${BASE} python -m rfm.cli.train_suiren_fusion_property ...
```

校验原则：

```text
只检查当前流程会读取和依赖的字段；
不为了历史残留额外扫描不会被当前流程读取的字段。
feature cache 复用前必须验证 reaction_id 顺序与当前 HDF5 split 一致；
atom-level cache 还必须验证 atom_map_order 和 n_atoms 一致。
```

## 3. Current Input Rules

Masked edit pretraining:

```text
external raw input: Z, BO_R, BO_P
2D training-time tokens: Z, BO_R, BO_P_visible, Delta_BO_visible, visibility
3D training-time tokens: Z, BO_R, BO_P_visible, Delta_BO_visible, visibility, D_R_irc
```

Reaction property regression:

```text
external raw input, 2D: Z, BO_R, BO_P
derived 2D basis: Delta_BO, abs_Delta_BO, A_R/A_P, Delta_A, formed/broken/order_changed
external raw input, IRC-3D: Z, BO_R, BO_P, D_R_irc, D_P_irc
derived IRC-3D basis: Delta_D, abs_Delta_D
```

Suiren fusion property regression:

```text
matched Stage B input/basis
optional frozen Suiren features as unified Reaction Encoder input features:
  suiren_2d_graph_features -> 2D graph projection -> initial reaction_token
  suiren_3d_graph_features -> 3D graph projection -> initial reaction_token
  suiren_2d_atom_features  -> 2D atom projection  -> atom_tokens, aligned by atom_map_order
  suiren_3d_atom_features  -> 3D atom projection  -> atom_tokens, aligned by atom_map_order
first version does not add Suiren pair tokens
train split uses forward + reverse augmentation; reverse swaps h_R/h_P and recomputes Delta_h
```

Stage C 正式实现使用 `UnifiedReactionInputAdapter`：Suiren features 和原始
2D/3D features 一起编码成同一个 `RFMEncoderInput`，再进入同一个 Reaction Encoder。
`2D+3D` 组合通过同时激活 2D 与 3D 输入通道实现，不把两类 cache 预先拼接成单一路
feature。Suiren 原始维度到 `hidden_dim` 的 projection 是输入编码层，不是 encoder
后处理模块。

active processed HDF5 只写 IRC R/P endpoint 坐标，不写 `coordinates_TS`。
`B3LYPD3_TZVP_IRC` 只提供 trajectory；R/P 坐标必须通过 `jump_idx` endpoint candidates
和 BO_R/BO_P graph-alignment audit 生成。
不再默认使用旧的 R/P atom numeric、芳香性、ring membership、degree、formal charge 等 RDKit 后处理特征。
若冻结 Suiren 2D 接口要求这些输入列，只允许写成常量 compatibility slots，不作为样本特异输入。
`core_atom` 和 `changed_pair` 作为训练标签时，统一从 `Delta_BO = BO_P - BO_R` 派生。
Suiren cache 训练/评估前必须验证 `failed.sum() == 0`；failed row 不能作为合法零特征进入模型。
Reaction Encoder 输出接口固定为 `atom_h: [B,N,H]`、`pair_h: [B,N,N,H]`、
`reaction_h: [B,H]`；property head / masked-edit heads 不随 Suiren feature 组合改变输入维度。

完整 MRTO 实现见 `docs/MRTO_FULL_ARCHITECTURE.md`。`mrto_full` 使用一套共享的
EquiformerV2 endpoint adapter 处理 R/P 独立坐标系，在主干内保持严格 R/P swap
even/odd fields，并通过 competitive sparse event slots 读写 atom/pair fields。Stage C
的 Suiren atom/graph 特征在每层算子内部作 parity-aware input conditioning，不是输出旁路。

## 4. Checkpoint Rule

Masked edit pretraining 和 reaction property regression 的 best checkpoint 都按 validation loss 最小选择：

```text
checkpoint selection = minimum valid_loss
```

默认训练配置：

```text
epochs = 100
early_stop_patience = 10
batch_size = 64 per GPU
optimizer = AdamW
lr = 2e-4
weight_decay = 1e-4
hidden_dim = 128
layers = 3
dropout = 0.1
gradient clipping = 5.0
```
