# Atom-Level Suiren Feature Schema

更新时间：2026-06-12

本文档定义当前 RFM / Reaction-QM 使用的 frozen Suiren atom-level feature cache。
Suiren encoder 固定不训练，不使用 LoRA，不使用 TS generation。所有 cache 写入远程
工作目录 `outputs/suiren_fusion/`，不能写入原始数据目录。

Atom-level Suiren cache rows 是给统一 Reaction Encoder input adapter 使用的
atom-map-aligned input features。它们不是模型输出，不是 post-encoder residual
features，也不能改变 Reaction Encoder 的 `atom_h / pair_h / reaction_h` 输出维度。

## 文件格式

full-scale atom-level cache 使用 HDF5 ragged schema：

```text
outputs/suiren_fusion/features_irc_rp_suiren_2d_atom/suiren_2d_atom_{split}_full.h5
outputs/suiren_fusion/features_irc_rp_suiren_3d_atom/suiren_3d_atom_{split}_full.h5
```

训练 split 可以先写 shard，再 merge 为 `*_train_full.h5`。评估只使用 forward-only
valid/test cache。

## 必需字段

| Key | Shape | Dtype | 含义 |
|---|---:|---|---|
| `reaction_ids` | `[n_samples]` | UTF-8 string | 与 source split 顺序一致的 reaction id |
| `n_atoms` | `[n_samples]` | int32 | 每个 reaction 的 atom-map-aligned 原子数 |
| `atom_ptr` | `[n_samples + 1]` | int64 | ragged atom feature 指针 |
| `atom_map_order` | `[sum_atoms]` | int32 | 每个 atom row 对应的 atom map id |
| `features` | `[sum_atoms, d]` | float32/float16 | Suiren atom embedding 组合；作为 atom-token input projection 的输入，不做 molecular pooling |
| `failed` | `[n_samples]` | bool | sample-level extraction failure flag |
| `processed` | `[n_samples]` | bool | streaming 写入完成标记 |
| `attrs.metadata_json` | JSON | string | source、modality、layout、geometry provenance |

## Feature Layout

2D atom cache：

```text
H_R_2d_atom
H_P_2d_atom
Delta_H_2d_atom = H_P_2d_atom - H_R_2d_atom
abs_Delta_H_2d_atom = |H_P_2d_atom - H_R_2d_atom|
```

3D atom cache：

```text
H_R_3d_atom
H_P_3d_atom
Delta_H_3d_atom = H_P_3d_atom - H_R_3d_atom
abs_Delta_H_3d_atom = |H_P_3d_atom - H_R_3d_atom|
```

当前 dim 为 `4 * 256 = 1024`。若未来 Suiren hidden size 改变，dim 以
`metadata_json.feature_layout` 和 `features.shape[1]` 为准。

## 对齐约束

1. `reaction_ids[i]` 必须等于 source split 第 `i` 条样本的 `reaction_id`。
2. `n_atoms[i]` 必须等于 source `len(atomic_numbers)`。
3. `features[atom_ptr[i]:atom_ptr[i+1]]` 的 atom 行顺序必须等于 source
   `atom_map_order` 顺序。
4. `atom_map_order[atom_ptr[i]:atom_ptr[i+1]]` 必须等于 source `atom_map_order`。
5. `atom_map_id` 只能用于构建/验证对齐，不能作为模型输入 embedding。
6. `reaction_core_mask` 若在分析中使用，只能作为训练标签或 grouped analysis target，
   不参与 Suiren atom feature cache 或 primary input token。

## Cache 完整性

队列脚本只有在以下条件全部满足时才允许跳过已有 cache：

```text
文件非空
processed.all() == true
features 行数与 atom_ptr[-1] 一致
atom_map_order 行数与 atom_ptr[-1] 一致
failed 长度与 reaction_ids 长度一致
failed.sum() == 0 for active training/evaluation cache
features 全部为 finite
```

不满足时，该 shard/cache 必须删除并重建，不能把 streaming 中断产生的 HDF5 当作完成。
如果 Suiren feature 生成失败，对应样本不能以零特征形式参与训练；应修复 cache，或显式
生成经过审计的 filtered split/cache，并让 matched baseline 使用同一过滤集合。

## 3D 限制

3D atom cache 只能使用 IRC R/P endpoint coherent R/P geometry。
smoke test 已显示 Suiren-Base 3D atom extraction 使用 AMP 会产生非有限 atom features；
当前 3D atom extraction 必须使用 no-AMP float32 inference。
