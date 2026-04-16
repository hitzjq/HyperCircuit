# 0416 实验总结

## 概述

本轮共 **10 个实验**（8 短线 + 2 长线），测试 5 个新变量：
1. Self-Attention RoPE（PG 位置编码）
2. condition_mode（full_trm vs embedding_only）
3. lora_r=4（极低秩）
4. 移除 head LoRA（lm_head/q_head）
5. lr 5e-6 vs 1e-5

## 代码改动

本轮涉及 3 个文件的改动（已完成）：

| 文件 | 改动 |
|------|------|
| `models/hypernetwork/pg_transformer.py` | 新增 `use_rope` 参数，可选 1D RoPE 注入 self-attention |
| `models/hypernetwork/__init__.py` | `ParameterGenerator` 透传 `use_rope` 到 PGTransformer |
| `meta_train.py` | 新增 `pg_use_rope`、`head_lora` 配置项；`head_lora=False` 时过滤并还原 lm_head/q_head |

## 短线实验（8 组，20k epochs）

### 全参数表

| # | 实验名 | RoPE | cond_mode | r | α | α/r | lr | wd | pg_b | pg_d | pg_acc | head_lora | dropout | bs | epochs |
|---|--------|:---:|-----------|:-:|:-:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R1 | rope_r8_full | **✅** | full_trm | 8 | 16 | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R2 | rope_r8_emb | **✅** | **emb_only** | 8 | 16 | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R3 | rope_r4_full | **✅** | full_trm | **4** | **8** | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R4 | rope_r4_emb | **✅** | **emb_only** | **4** | **8** | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R5 | norope_r8_emb | ❌ | **emb_only** | 8 | 16 | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R6 | rope_r8_lr5e6 | **✅** | full_trm | 8 | 16 | 2 | **5e-6** | 0.1 | 2 | 256 | 4 | ✅ | 0 | 2048 | 20k |
| R7 | rope_r8_nohead | **✅** | full_trm | 8 | 16 | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | **❌** | 0 | 2048 | 20k |
| R8 | rope_r8_nohead_emb | **✅** | **emb_only** | 8 | 16 | 2 | 1e-5 | 0.1 | 2 | 256 | 4 | **❌** | 0 | 2048 | 20k |

> 加粗项为与基线 R1 不同的参数。

### 对比关系

```
R1 vs C1(上轮)     → RoPE 是否有效
R1 vs R2            → full_trm vs embedding_only（有 RoPE）
R2 vs R5            → RoPE 在 emb_only 下是否有用
R1 vs R3            → r=8 vs r=4
R3 vs R4            → r=4 下 full_trm vs emb_only
R1 vs R6            → lr=1e-5 vs 5e-6
R1 vs R7            → 移除 head LoRA 的效果
R7 vs R8            → 无 head LoRA 下 full_trm vs emb_only
```

### 脚本分配

| 脚本 | 实验 |
|------|------|
| `run_0416_1.sh` | R1 + R2 |
| `run_0416_2.sh` | R3 + R4 |
| `run_0416_3.sh` | R5 + R6 |
| `run_0416_4.sh` | R7 + R8 |

## 长线实验（2 组，100k epochs）

| # | 实验名 | RoPE | cond_mode | r | α | lr | wd | epochs | warmup | eval_interval |
|---|--------|:---:|-----------|:-:|:-:|:---:|:---:|:---:|:---:|:---:|
| L1 | long_r8_norope | ❌ | full_trm | 8 | 16 | 1e-5 | 0.1 | 100k | 5000 | 5000 |
| L2 | long_r8_rope | **✅** | full_trm | 8 | 16 | 1e-5 | 0.1 | 100k | 5000 | 5000 |

**选择理由**：
- L1: 上轮最优配置（C1 r=8），直接延长到 100k 看长线是否继续上升
- L2: 加 RoPE 后的同一配置，用长线验证 RoPE 的持续效果
- 两者唯一区别是 RoPE on/off，对比干净

| 脚本 | 实验 |
|------|------|
| `run_0416_long1.sh` | L1 |
| `run_0416_long2.sh` | L2 |

## 参考基线

| 来源 | pass@2 | 配置 |
|------|:---:|------|
| 基座 step_518071（无 LoRA） | 0.4437 | — |
| 上轮最优 C1 (best epoch) | 0.4637 | r=8, 无 RoPE, full_trm, lr=1e-5 |
| 上轮最优 C2 (best epoch) | 0.4625 | r=32, 无 RoPE, full_trm, lr=1e-5 |

## 日志输出

所有日志保存到 `logs/logs0416/`。
