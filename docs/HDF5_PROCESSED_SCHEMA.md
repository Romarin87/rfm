# RFM Processed HDF5 Schema

更新时间：2026-06-12

本文档定义当前 RFM / Reaction-QM full-scale processed dataset 的主保存格式。原始
Reaction-QM 数据目录只读；所有 processed HDF5 必须写在远程工作目录
`outputs/processed/` 下。下列路径是重新 bootstrap 后的规划产物路径；远程工作区为空时
必须从官方只读源重新构建，不能假设任何 HDF5、cache 或 manifest 已存在。

## 适用产物

| Dataset | Path |
|---|---|
| full IRC R/P endpoint B3LYP | `outputs/processed/full_b3lyp_irc_rp/b3lyp_delta_full_irc_rp.h5` |
| full IRC R/P endpoint train forward+reverse B3LYP | `outputs/processed/full_b3lyp_irc_rp_aug_reverse/b3lyp_delta_full_irc_rp_forward_reverse.h5` |

## 顶层结构

```text
/
  attrs:
    schema_version = "rfm_reaction_delta_hdf5_v0.1"
    metadata_json  = JSON string
  splits/
    train/
    valid/
    test/
```

`metadata_json` 记录 source CSV/H5 路径、处理模式、生成时间、只读源数据约束和 leakage 约束。
当 `mode=irc_rp` 时，`metadata_json` 还必须记录 `source_irc_h5`、
`irc_rp_protocol=true`、`irc_rp_assignment_reference=BO_R/BO_P graph alignment`。

## Split Group

每个 `splits/{split}` group 包含：

| Key | Shape / Type | 用途 |
|---|---|---|
| `reaction_id` | `[N]` UTF-8 string | reaction id |
| `sample_json` | `[N]` UTF-8 string | 完整 `ReactionDeltaSample` JSON payload，用于当前训练、审计和特征缓存 loader |
| `n_atoms` | `[N] int32` | atom-map 对齐后的原子数 |
| `n_reactants` | `[N] int16` | reactant fragment 数 |
| `n_products` | `[N] int16` | product fragment 数 |
| `targets` | `[N, 6] float32` | `dE, dE_dagger, dH, dH_dagger, dG, dG_dagger` |
| `atomic_numbers_ptr` | `[N+1] int64` | packed `atomic_numbers` 指针 |
| `atomic_numbers` | `[sum_n] int16` | packed atomic numbers |
| `atom_map_order_ptr` | `[N+1] int64` | packed `atom_map_order` 指针 |
| `atom_map_order` | `[sum_n] int32` | packed atom-map ids，仅用于 R/P 对齐 |
| `reaction_core_mask_ptr` | `[N+1] int64` | packed core mask 指针 |
| `reaction_core_mask` | `[sum_n] int8` | 可选训练标签 / analysis target；具体任务是否使用由 task definition 决定 |
| `has_coordinates_R/P` | `[N] bool` | IRC R/P endpoint 坐标模态是否存在 |
| `coordinates_R/P` | `[N, max_atoms, 3] float32` | 若该 split 全部样本有对应坐标则写入；无效 padding 为 NaN |

active processed HDF5 不写 `coordinates_TS`。TS 只以能量标签中的 barrier 信息间接存在；
Stage A/B/C 主线不能读取真实 TS geometry。

## IRC R/P Metadata

`B3LYPD3_TZVP_IRC.h5` 只提供 official IRC trajectory，不直接提供可作为模型输入的
R/P 坐标字段。`mode=irc_rp` 的 builder 必须先从 trajectory 中执行统一 endpoint
protocol，再把定向后的 `coordinates.R/P` 写入 `sample_json` 和 packed 坐标数组。

`mode=irc_rp` 的样本必须在 `sample_json.coordinate_metadata` 中记录：

| Key | 含义 |
|---|---|
| `endpoint_protocol` | 固定为 `irc_rp_from_jump_idx_last_candidates` |
| `endpoint_candidate_A` | `frames[jump_idx]` |
| `endpoint_candidate_B` | `frames[-1]` |
| `endpoint_assignment_reference` | 固定为 `BO_R/BO_P distance-threshold graph alignment` |
| `endpoint_assignment_factor` | 共价半径阈值系数，当前为 `1.25` |
| `endpoint_assignment` | `A=R,B=P` 或 `A=P,B=R` |
| `assignment_cost_forward_A_R_B_P` | candidate_A/B 作为 R/P 时，对 `BO_R/BO_P` adjacency 的 FP+FN mismatch count |
| `assignment_cost_reverse_A_P_B_R` | candidate_A/B 作为 P/R 时，对 `BO_P/BO_R` adjacency 的 FP+FN mismatch count |
| `assignment_cost_unit` | 固定为 `upper-triangle adjacency FP+FN count` |
| `assignment_margin` | 两种 graph-alignment cost 的差值；越大说明 R/P 判定越稳定 |
| `jump_idx` | `argmax(E[t+1]-E[t])` |
| `post_jump_ts_idx` | `jump_idx + 1`，TS-near restart frame，只用于审计，不作为 R/P 主线输入 |
| `max_positive_jump_ev` | `E[jump_idx+1] - E[jump_idx]` |
| `ts_coordinates_stored` | 固定为 `false` |
| `raw_n_frames` / `raw_n_energies` | 原始 IRC frame / energy 数量 |
| `frame_energy_length_trimmed` | frame 和 energy 长度不一致时是否按最短长度裁剪 |
| `trimmed_n` | 若裁剪，实际使用的 frame/energy 数量 |

manifest / report 必须记录全局和 split-level 的：

```text
endpoint_assignment_counts
endpoint_boundary_counts:
  max_positive_jump_ev_le_0
  assignment_margin_lt_1e-4
  assignment_margin_lt_1e-3
  assignment_margin_min
```

旧 `frames[0]` / `frames[-1]` endpoint protocol 已废弃，不能作为 HDF5 构建模式保留。

## Sample JSON Payload

`sample_json` 必须保留现有训练代码需要的字段：

```text
reaction_id
source_dataset
level_of_theory
split
reaction_smiles
atom_map_order
atomic_numbers
n_reactants
n_products
bonds_R
bonds_P
changed_pairs
reaction_core_atom_maps
reaction_core_mask
coordinates
modality_mask
coordinate_metadata
targets
audit
```

`audit` 是样本级质检摘要，不作为模型输入。当前只记录现行构建流程实际使用和报告的
质检项，例如：

```text
source_2d_graph
r_p_atom_map_z_match
ts_atom_order_matches_atom_map_order
irc_atomic_numbers_match
irc_rp_endpoint_protocol
irc_rp_assignment_reference
irc_rp_endpoint_assignment
irc_rp_assignment_margin
irc_rp_boundary_flags
```

## 约束

- `atom_map_order` 和 atom map id 只能用于 R/P 对齐，不能作为模型输入 embedding。
- `reaction_core_mask` 若被任务使用，只能作为训练标签或 grouped analysis target，不进入主输入 token。
- active HDF5 不写 `coordinates.TS` / `coordinates_TS`；Stage A/B/C 主线不把 TS geometry 用作模型输入。
- JSON/CSV/Markdown 仅用于小型审计与复盘文件；full processed reaction samples 的主产物是 HDF5。
- 当前正式训练入口应使用 `file.h5::train`、`file.h5::valid`、`file.h5::test` 路径。
- forward+reverse augmented HDF5 只允许 `train` split 增强；`valid/test` 必须保持 forward-only。

## 必跑审计

重建 `full_b3lyp_irc_rp` 后必须先完成 graph-alignment audit
和 HDF5 consistency audit。审计必须验证：

- 从官方 IRC HDF5 重新计算 endpoint candidate 与 R/P assignment；
- packed `coordinates_R/P` 与审计得到的 IRC R/P endpoint assignment 一致；
- `reaction_id`、`atom_map_order`、`BO_R/BO_P`、`D_R_irc/D_P_irc` 的 shape 与语义一致；
- active HDF5 不包含 TS geometry 输入字段。
