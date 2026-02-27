# TRM (TinyRecursiveModels) "无遗憾" LoRA 微调执行方案

**项目负责人**: 张峻齐
**目标模型**: TRM (~7M params, arXiv:2510.04871)
**代码仓库**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
**LoRA指导意见**: https://thinkingmachines.ai/blog/lora/

## 1. 方案背景与目标
本方案旨在对 ARC-AGI 推理模型 TRM 进行无性能折损的 LoRA 微调。依据 "LoRA Without Regret" 的最新研究见解，我们将抛弃仅在 Attention 层挂载 LoRA 的传统设定，转而采用**全线性层覆盖（All-Linear）**与**高学习率**的策略，以期在极低显存占用下达到全量微调（FullFT）的推理泛化水平。

## 2. 核心超参数设定准则
请 Antigravity 在构建训练脚本时，严格遵循以下四条核心原则：

* **Target Modules (全线性层覆盖)**：TRM 是一个小参数的递归架构，其表达能力极度依赖其内部的投影层和前馈网络。必须在 PEFT 配置中动态识别并覆盖所有的 `nn.Linear` 模块，绝对不能漏掉类似 MLP 层的参数。
* **Learning Rate (10倍倍率)**：LoRA 的最佳学习率不应与 FullFT 相同。请将优化器的初始学习率设定为 TRM 默认微调学习率的 **10倍** 左右（建议初始搜索空间定在 `1e-3` 到 `5e-3`），并配合 Cosine Decay 调度器。
* **Rank (r) 设定**：ARC-AGI 任务具备极高的逻辑复杂度和数据密度。推荐使用较高的 Rank 值（如 `r=128` 或 `r=256`），并保持缩放比例稳定（即设置 `alpha = r` 或 `alpha = 2 * r`）。
* **Batch Size**：采用中等规模的 Batch Size（如 32 - 128）。由于 TRM 会在 latent 空间进行多次递归（unrolled steps），过大的 Batch Size 容易在反向传播时引发梯度震荡。

