# Circuit 特征与 Hypernet 集成方案

## 一、背景概述

我们现在的目标是将提取到的 `circuit` 作为 `condition` 的一部分喂给 `hypernet`，配合原有的 `query embedding` 让 `hypernet` 生成这道题的 lora。

由于我们要跑很多的 epochs，如果在训练过程中在线提取 `circuit`，每一步都要通过完整的 CodeCircuit pipeline (收集激活、VJP、构图、跑完前几个步骤等) 会极大地拖慢训练速度。

**因此，我们的核心思路是“化在线为离线”：**
既然对于同一个 query (同样的 inputs)，使用基础的预训练模型提取到的 `circuit` 是一模一样、始终不变的，我们就可以预先对训练集和测试集所有的 query 跑一遍提取，把最终的 53 维特征向量存下来。训练的时候，就像加载数据一样，直接根据当前的 query 把存好的 `circuit` 特征读出来，拼接给 `hypernet` 即可。

---

## 二、当前数据流梳理

### 2.1 离线提取的产出

运行 `run_pipeline_prod.sh` 后，产生的文件 `cc_advanced_features.pt` 包含：

```python
{
    "features": torch.Tensor,        # shape: (n_queries, 53)，每个 query 的电路特征
    "query_mapping": [                # 长度 n_queries 的 list
        {
            "graph_file": "graph_000000.pt",
            "graph_index": 0,
            "set_name": "all",
            "puzzle_identifiers": tensor([42]),
            "inputs": tensor([[3, 5, 0, ...]]),   # 该 query 的确切输入序列
            "labels": tensor([[7, 2, 0, ...]]),
        },
        ...
    ],
    "feature_dim": 53,
    "n_queries": 总数
}
```

### 2.2 Hypernet 当前输入与目标

**现在：**
`ParameterGenerator` 的输入是 `z_H` (来自基座模型的特征或 embedding)，被映射到 256 维的 `cond`。

**目标：**
我们不仅要传 `z_H`，还要传 53 维的离线 `circuit` 特征。需要在网络内把这 53 维映射到能够和 `cond` 结合的维度。

---

## 三、核心设计方案

### 3.1 怎么从离线特征库匹配到当前 DataLoader 里的 Query？

**问题**：即使是离线提取好，我们在 DataLoader 给出一个 batch 数据时，怎么知道这个 batch 里的 sample 分别对应 `cc_advanced_features.pt` 里的哪一行？`puzzle_identifiers` 不能用，因为同一个 puzzle 数据增强后有不同的 `inputs`（比如旋转、翻转）。

**解决方案：基于 inputs 序列的 MD5 Hash 构建快速查找表**

因为每个 example 都有唯一的 `inputs` 序列，我们可以：
1. **加载期**：在 `meta_train.py` 开始时，读入 `cc_advanced_features.pt`，用里面每个 mapping 的 `inputs` 算出一个 MD5 字符串。建立一个字典 `lookup_table[md5] = idx`。
2. **训练期**：拿出一个 batch 的时候，对 batch 里的每一行（每一个 query 的 `inputs`）也算一遍 MD5，去 `lookup_table` 里面查。查到的 index 就是它的 53 维特征。

*这样做完全模拟了“在线对这个 query 算一遍”，因为我们是通过极其具体的输入序列去配对的。*

```python
import hashlib

def inputs_hash(inputs_tensor):
    return hashlib.md5(inputs_tensor.cpu().numpy().tobytes()).hexdigest()
```

### 3.2 维度怎么和 condition 拼在一起？

离线取到了 `(B, 53)` 的电路特征，怎么喂进 `Hypernet`？

**解决方案：**
1. 在 `ParameterGenerator` 中，加一个针对 `circuit` 的投影网络 `circuit_proj`。
2. 这个投影网络（两层 Linear 加 SiLU）也是**可以训练**的，随着 Hypernet 一起更新梯度。它会把 53 维映射到 `d_model` (256 维)。
3. 把投影后的 `(B, 1, d_model)` 作为一个额外的“全局电路 Token”，用 `torch.cat` 拼在原本 S 个序列长度的 `cond` 前面。
4. PG Transformer 的 Cross-Attention 原本是 Key/Value 序列长度为 S，现在变成 S+1，无缝兼容，不需要改核心注意力机制。

---

## 四、具体代码修改计划

我们将修改两个文件：

### 4.1 修改 `models/hypernetwork/__init__.py`

```python
class ParameterGenerator(nn.Module):
    def __init__(
        self,
        # ... 原本的参数 ...
        circuit_dim: int = 0,  # 新增：告诉模型我们是不是开启了 circuit 这个功能
    ):
        super().__init__()
        # ... 原有代码保留不变 ...

        # 新增：定义可训练的映射网络，把 53 维映射到超网里
        self.use_circuit = circuit_dim > 0
        if self.use_circuit:
            self.circuit_proj = nn.Sequential(
                CastedLinear(circuit_dim, self.d_model, bias=False),
                nn.SiLU(),
                CastedLinear(self.d_model, self.d_model, bias=False),
            )

    def forward(self, z_H, scale=2.0, circuit_feat=None):
        B = z_H.shape[0]

        # 原本的映射
        cond = self.cond_proj(z_H)  # [B, S, d_model]

        # 新增的拼接操作模拟
        if self.use_circuit and circuit_feat is not None:
            circuit_token = self.circuit_proj(circuit_feat) # [B, d_model]
            circuit_token = circuit_token.unsqueeze(1)      # [B, 1, d_model]
            cond = torch.cat([circuit_token, cond], dim=1)  # 拼起来！变成 [B, S+1, d_model]

        # 后面原路输出
        # ...
```

### 4.2 修改 `meta_train.py`

1. **Config 中加字段**：增加一个 `circuit_features_path` 参数。
2. **初始化**：在脚本跑起来的时候，如果给了提取好的 circuit 特征，在所有 epoch 开始之前，把他们全部读进内存（反正才几十M），并构建好 `lookup_dict`。
3. **在 `train_batch` / `evaluate`** 中：获取 batch 的前一刻，把 `batch['inputs']` 丢进我们写好的找 circuit 特征的辅助函数，拿到一个 `(B, 53)` 的 tensor `circuit_feat`。
4. 传给 PG 模型：`pg_model(z_H, ..., circuit_feat=circuit_feat)`

---

## 五、方案优势与一致性确认

正如你强调的，这套方案在概念上完全等价于：**“假设在训练的每一刻我们都在线算一次完整的电路，然后把它拼到了条件序列中”**。

- 这个离线查找的做法确保了结果绝不会搞混（通过最原始输入数据的 hash 对齐），时间消耗 O(1) 查表。
- 53 维到 256 维的映射在 `circuit_proj` 里发生了梯度的可训练更新，保证了我们可以教会 hypernet 如何理解和利用这个提取来的电路统计量。

唯一需要在跑训练命令前注意的：
你需要提前提取好**所有会参与训练和测试**的数据集。也就是说如果你的 `dataset.json` 里有 1000 个 augmented 数据在训练，这 1000 个数据都必须在交给我的 `cc_advanced_features.pt` 文件里。找不到的我目前打算就用 0 补上。
