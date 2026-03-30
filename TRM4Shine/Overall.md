# TRM (TinyRecursiveModels) "无遗憾" LoRA 微调执行方案

**项目负责人**: 张峻齐
**目标模型**: TRM (~7M params, arXiv:2510.04871)
**代码仓库**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
**LoRA指导意见**: https://thinkingmachines.ai/blog/lora/

## 1. 方案背景与目标
本方案旨在对 ARC-AGI 推理模型 TRM 进行无性能折损的 LoRA 微调。依据 "LoRA Without Regret" 的最新研究见解，我们将抛弃仅在 Attention 层挂载 LoRA 的传统设定，转而采用**全线性层覆盖（All-Linear）**与**高学习率**的策略，以期在极低显存占用下达到全量微调（FullFT）的推理泛化水平。

## 2. 核心超参数设定准则
请 Antigravity 在构建训练脚本时，严格遵循以下四条核心原则：

* **Target Modules (全线性层覆盖)**：TRM 是一个小参数的递归架构，其表达能力极度依赖其内部的投影层和前馈网络，必须覆盖所有的线性层，绝对不能漏掉 MLP 层的参数。
  > ⚠️ **重要**：TRM 内部的线性层全部使用自定义的 `CastedLinear`（见 `models/layers.py`），而**非**标准的 `nn.Linear`。`peft` 库的自动识别机制无法感知 `CastedLinear`，直接套用会造成「代码不报错但 LoRA 实际上没有挂上」的静默失败。因此，**不使用 `peft`**，改为手写一个 `inject_lora()` 函数，主动遍历并将所有 `CastedLinear` 替换为 `LoRACastedLinear`（即 $W + \frac{\alpha}{r} AB$）。
* **Learning Rate (10倍倍率)**：LoRA 的最佳学习率不应与 FullFT 相同。请将优化器的初始学习率设定为 TRM 默认微调学习率的 **10倍** 左右（建议初始搜索空间定在 `1e-3` 到 `5e-3`），并配合 Cosine Decay 调度器。
* **Rank (r) 设定**：ARC-AGI 任务具备极高的逻辑复杂度和数据密度，需要足够大的 Rank。推荐使用 **`r=64`**，`alpha=32`。对于 TRM 约 256~512 的隐藏维度，`r=64` 已能提供接近全量微调（FullFT）的表达容量，同时保持 LoRA 参数量远小于原始权重（`r=128/256` 在此量级下 A×B 的规模将接近原矩阵本身，失去轻量意义）。
* **Batch Size**：采用中等规模的 Batch Size（如 32 - 128）。由于 TRM 会在 latent 空间进行多次递归（unrolled steps），过大的 Batch Size 容易在反向传播时引发梯度震荡。

