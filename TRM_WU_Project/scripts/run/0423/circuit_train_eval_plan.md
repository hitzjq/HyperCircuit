# 0423 Circuit HyperNet 训练与延后评估方案

## 目标

当前只有训练集 circuit 特征已经准备好，测试集 circuit 特征还没有准备好。目标是先用 train circuit 训练一批 20k 短线 checkpoint，等 test/all circuit 准备好后，再统一评估所有中间 checkpoint 和最终 checkpoint。

本方案已落地训练阶段的最小改动与运行脚本：

- `meta_train.py` 增加 `skip_eval`，用于跳过 test evaluation 但保留 checkpoint 保存。
- `config/cfg_wu4trm.yaml` 增加 `skip_eval: False` 默认值，不影响旧实验默认行为。
- `scripts/run/0423/run_0423_circuit_train.sh` 跑 4 套 20k circuit 短线训练配置。

## 已有 circuit 输入

训练阶段先使用服务器上的 train-only circuit 文件：

```bash
/mnt/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/prod_0421_1742/cc_advanced_features_train.pt
```

这个路径应作为 `circuit_features_path` 传给 `meta_train.py`。该文件应包含：

- `features`: shape 约为 `[n_train_queries, 53]`
- `query_mapping`: 每个 query 的 `inputs` 映射信息
- `feature_dim`: 53
- `n_queries`

当前 `meta_train.py` 会用 batch 内每条 `inputs` 的 MD5 去 `query_mapping` 中查 circuit 特征。训练阶段只要 batch 来自 train split，这份 `cc_advanced_features_train.pt` 就是正确输入。

## circuit encoder 是否可训练

是可训练的。

`ParameterGenerator` 中的 circuit 投影为：

```python
self.circuit_proj = nn.Sequential(
    CastedLinear(circuit_dim, d_model, bias=False),
    nn.SiLU(),
    CastedLinear(d_model, d_model, bias=False),
)
```

当 `circuit_features_path` 非空时，`circuit_dim > 0`，`circuit_proj` 会被创建为 PG 模型的一部分。优化器包含 `list(bundle["pg"].parameters())`，因此 `circuit_proj` 的两个 `CastedLinear` 权重会随 HyperNet 一起训练。

## 为什么训练期要跳过 eval

如果训练时只传 `cc_advanced_features_train.pt`，那么 `evaluate()` 跑 test split 时，test batch 的 `inputs` 在 train-only circuit 文件中找不到，当前逻辑会用全 0 circuit 特征兜底。

这会导致：

- 训练中间 eval 指标不可信
- 日志中产生大量无意义测试结果
- 浪费 H200 时间

因此训练阶段应只训练和保存 checkpoint，不做 test evaluation。等 test/all circuit 准备好后，再统一 evaluate。

## 已落地的最小代码改动

已在 `meta_train.py` 中新增一个配置开关：

```yaml
skip_eval: true
```

训练循环在到达 `eval_interval` 时：

- 如果 `skip_eval=false`，保持原逻辑：evaluate，然后保存 checkpoint
- 如果 `skip_eval=true`，跳过 evaluate，但仍然调用 `save_train_state(...)` 保存 checkpoint

这样 `eval_interval` 在训练期就等价于 `checkpoint_interval`。

伪逻辑：

```python
if _iter_id >= config.min_eval_interval:
    if config.skip_eval:
        if rank == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)
        continue

    metrics = evaluate(...)
    if rank == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
        save_train_state(config, train_state_eval)
```

这个改动的好处是：

- 不影响默认训练行为
- 不需要 test circuit 就可以保存所有中间 checkpoint
- 之后 test circuit 完成后可以补跑正式评估

## 正式评估阶段

等 test circuit 或 all circuit 准备好后，正式评估时应使用包含 test query 的 circuit 文件，例如：

```bash
/mnt/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/<prod_run>/cc_advanced_features.pt
```

或：

```bash
/mnt/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/<prod_run>/cc_advanced_features_all.pt
```

正式评估需要遍历每个实验目录下的所有 `step_*.pt`，包括：

- 中间 checkpoint
- 最终 checkpoint

建议输出统一汇总表：

```text
run_name, step, epoch, ARC/pass@1, ARC/pass@2, ARC/pass@5, ARC/pass@10, exact_accuracy, lm_loss
```

后续需要补一个 eval-only 脚本或 eval-only 模式，用于加载 `save_train_state()` 保存的联合 checkpoint：

```python
{
    "trm": ...,
    "pg": ...,
    "step": ...
}
```

并在加载后只跑 `evaluate()`，不继续训练。

## 20k 短线训练设置

建议第一轮 circuit 短线训练使用：

```bash
epochs=20000
eval_interval=2000
checkpoint_every_eval=True
skip_baseline_eval=True
skip_eval=True
global_batch_size=2048
lr_warmup_steps=2000
circuit_features_path=/mnt/kbei/HyperCircuit/TRM_WU_Project/CodeCircuit_TRM_Arc1/runs/prod_0421_1742/cc_advanced_features_train.pt
```

在 `skip_eval=true` 的计划下，`eval_interval=2000` 实际作为 checkpoint 保存间隔使用。每套配置会保存：

- epoch 2000
- epoch 4000
- epoch 6000
- epoch 8000
- epoch 10000
- epoch 12000
- epoch 14000
- epoch 16000
- epoch 18000
- epoch 20000

## 0414 候选配置

从现有日志和 0416 README 中记录的“上轮最优”看，0414 选择两套：

### 0414-C1: lorar8

历史表现：

- best `ARC/pass@2`: 0.46375
- best epoch: 14000

配置：

```bash
TAG="C1_lorar8_circuit"
condition_mode="full_trm"
pg_use_rope=False
head_lora=True
pg_num_blocks=2
pg_d_model=256
lora_r=8
lora_alpha=16
lr=1e-5
puzzle_emb_lr=1e-3
weight_decay=0.1
```

选择理由：

- 0414 短线历史最优
- 也是 0416 long1 的延长基础
- 参数规模适中，适合作为 circuit 版主基线

### 0414-C2: lorar32

历史表现：

- best `ARC/pass@2`: 0.46250
- best epoch: 16000

配置：

```bash
TAG="C2_lorar32_circuit"
condition_mode="full_trm"
pg_use_rope=False
head_lora=True
pg_num_blocks=2
pg_d_model=256
lora_r=32
lora_alpha=64
lr=1e-5
puzzle_emb_lr=1e-3
weight_decay=0.1
```

选择理由：

- 0414 第二强候选
- 与 C1 的核心差异是 LoRA rank
- 可以观察 circuit 条件是否更偏好高 rank 动态 LoRA

## 0416 候选配置

0416 选择两套 20k 短线配置，不选 long 配置作为第一轮训练。

### 0416-R8: rope_r8_nohead_emb

历史表现：

- best `ARC/pass@2`: 0.46500
- best epoch: 8000

配置：

```bash
TAG="R8_rope_r8_nohead_emb_circuit"
condition_mode="embedding_only"
pg_use_rope=True
head_lora=False
pg_num_blocks=2
pg_d_model=256
lora_r=8
lora_alpha=16
lr=1e-5
puzzle_emb_lr=1e-3
weight_decay=0.1
```

选择理由：

- 0416 短线历史最优
- embedding-only 训练速度更快
- no-head LoRA 降低输出头干扰，适合作为 circuit 条件的高性价比版本

### 0416-R2: rope_r8_emb

历史表现：

- best `ARC/pass@2`: 0.46375
- best epoch: 10000

配置：

```bash
TAG="R2_rope_r8_emb_circuit"
condition_mode="embedding_only"
pg_use_rope=True
head_lora=True
pg_num_blocks=2
pg_d_model=256
lora_r=8
lora_alpha=16
lr=1e-5
puzzle_emb_lr=1e-3
weight_decay=0.1
```

选择理由：

- 0416 第二强短线候选之一
- 与 R8 只差 `head_lora`
- 可以判断 circuit 条件下是否仍然应该移除 head LoRA

## 第一轮实验矩阵

| Source | Tag | condition_mode | RoPE | head_lora | r | alpha | lr | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0414 | C1_lorar8_circuit | full_trm | False | True | 8 | 16 | 1e-5 | 20000 |
| 0414 | C2_lorar32_circuit | full_trm | False | True | 32 | 64 | 1e-5 | 20000 |
| 0416 | R8_rope_r8_nohead_emb_circuit | embedding_only | True | False | 8 | 16 | 1e-5 | 20000 |
| 0416 | R2_rope_r8_emb_circuit | embedding_only | True | True | 8 | 16 | 1e-5 | 20000 |

## 预期输出目录

每个 run 使用独立目录：

```bash
checkpoints/WU4TRM_<TAG>_bs2048_8gpus_<MMDD_HHMM>/
logs/logs0423/WU4TRM_<TAG>_bs2048_8gpus_<MMDD_HHMM>.log
```

checkpoint 目录中应至少有：

- `step_<step>.pt`
- `all_config.yaml`
- 复制的模型源码文件

正式评估后另行输出：

```bash
logs/logs0423_eval/circuit_eval_summary.tsv
logs/logs0423_eval/<run_name>_step_<step>.log
```

## 并行运行脚本

为适配 4 台 8 卡机器，`scripts/run/0423/` 目录下已拆分为 4 个独立训练脚本：

- `run_0423_1_c1_lorar8.sh`
- `run_0423_2_c2_lorar32.sh`
- `run_0423_3_r8_nohead_emb.sh`
- `run_0423_4_r2_rope_r8_emb.sh`

建议一台机器运行一个脚本。原先的聚合脚本 `run_0423_circuit_train.sh` 仍然保留，主要用于单机串行补跑。

## 后续执行顺序

1. 已完成：最小改 `meta_train.py`，增加 `skip_eval`，训练期跳过 evaluate 但保留 checkpoint 保存。
2. 已完成：新建 0423 训练脚本，跑上述 4 套 20k 短线配置，传入 train circuit。
3. 待执行：等 test/all circuit 完成。
4. 待实现：新建或补充 eval-only 工具，遍历 4 个 run 的所有 `step_*.pt`。
5. 待执行：用 all/test circuit 正式 evaluate，生成统一汇总表。
6. 待决策：按 `ARC/pass@2` 和 `ARC/pass@5` 选出进入长线训练的候选。

## 风险与注意事项

- 训练期如果只使用 train circuit，任何 test eval 都不可信，应跳过。
- 如果 batch 中某条 train query 在 `cc_advanced_features_train.pt` 里找不到，会使用全 0 circuit 特征兜底；日志中应保留少量 warning 以便发现覆盖率问题。
- 0414 的 `full_trm` 配置更慢，0416 的 `embedding_only` 配置更适合快速验证。
- 之后正式评估必须使用包含 test query 的 circuit 文件，否则评估仍然会退化为 0 circuit。
