# ARC-AGI-1 实验全流程文件一览 (TRM)

要运行 `run_ARCAGI1.sh` 并且跑通这套代码，整个实验从原理上被分为了**两个截然不同的阶段**：**数据准备与增强阶段** 和 **模型实体预训练阶段**。

以下是所牵扯到的全部核心文件及其绝妙的分工：

---

## 阶段一：数据准备与增强 (Data Augmentation)
*必须先运行这些文件，才能把原始题库瞬间扩充千倍。*

1. **`dataset/build_arc_dataset.py`**
   - **功能**：增强车间的主控脚本。它负责读取原始的 `.json` 题目，控制 1000 倍的生成变种，用哈希算法防止变种重复，并在最后打平输出。
   - **核心机制**：它负责给变出的一百万道“全新题”，发那一百万张名叫 `puzzle_identifier` 的离散号码牌。

2. **`dataset/common.py`**
   - **功能**：增强算法工具箱。
   - **核心机制**：里面写明了网格矩阵是怎么进行 `dihedral_transform` （所谓的 8 种立体翻转和镜像操作）的物理代码。

3. **`data/.../` (生成的主输出目录)**
   - **功能**：存放了几百 MB 甚至 GB 级别的生成物。包含了被彻底压扁的 Numpy 数组（`inputs.npy`, `labels.npy` 等）。在这里，还能找到 `dataset.json`（存放了词表大小）以及 `identifiers.json`（字符串ID到数字号码牌的历史映射账本）。

---

## 阶段二：模型预训练网络 (Model Pre-training / 执行 `run_ARCAGI1.sh`)
*数据铺好后，运行 bash 脚本便进入此炼丹流程。*

### 1. 入口与总控配置
4. **`pretrain.py`**
   - **功能**：实验的“心脏”引擎。它负责启动分布式外骨骼（`torchrun`），根据配置搭建起数据喂粮管道，组合并管理 `train_state`（里面包裹了模型和优化器），并真正写出了 `for epoch` 的大循环去不断拉取数据前向传播。如果我们要写 LoRA，只需魔改这个文件。

5. **`config/cfg_pretrain.yaml`**
   - **功能**：最高指挥部的默认配置单 (Hydra 格式)。`lr`、`batch_size` 都在这里。bash 脚本里那句看似随意的 `arch.L_layers=2`，其实就是运行时越权覆盖了这个 yaml 里的默认值。

### 2. 数据搬运工
6. **`puzzle_dataset.py`**
   - **功能**：PyTorch 定制级装卸工。
   - **核心机制**：把硬盘里又大又碎的 `.npy` 数组，高效封包成一个个 Tensor Batch。且负责保证里面的内容（输入 `inputs`）、答案（`labels`）和其对应的题号牌（`puzzle_identifiers`）牢牢绑定绝不错位。

### 3. 模型车间 (微型 7M TRM 引擎)
7. **`models/recursive_reasoning/trm.py`**
   - **功能**：TRM 总装工厂。
   - **核心机制**：定义了 `TinyRecursiveReasoningModel_ACTV1`。这区区三百行代码是全篇论文的灵魂。它明确指出了“虚拟题号特征”是怎么拼接在序列车头的，并且用 Python 的最原生的 `for _ in range(cycles):` 语法，写出了大宏观 `H_cycles` 和执行级 `L_cycles` 是怎么疯狂自我递归互相打磨隐藏状态的。

8. **`models/layers.py`**
   - **功能**：底层零件加工厂。
   - **核心机制**：它并没有去调 PyTorch 现成的臃肿模块，而是纯手工打了类似于 `CastedLinear` 和 `SwiGLU` 这样带加速和降精度的定制零件。

9. **`models/sparse_embedding.py`**
   - **功能**：模型的外界“外挂记忆U盘”。
   - **核心机制**：存放着那张拥有约“一百万行”，用来通过 `puzzle_identifier` 题号查专属特征的长阵列查询字典。

### 4. 监工与判卷官
10. **`models/losses.py`**
    - **功能**：惩罚机制函数。除了常规交叉熵，专门实现了与 ACT “提前交卷”强相关的加强学习截断策略 Loss（Q-value 梯度）。

11. **`evaluators/arc.py`**
    - **功能**：地狱铁面判官。模型平时的 Loss 只是错一个像素罚一点，而到了验证环节调用这个评价器时，它只认 **“精确匹配算作答对全给，错一个像素计 0 分 (Exact Match Accuracy)”** 这种极致粗暴的判分。
