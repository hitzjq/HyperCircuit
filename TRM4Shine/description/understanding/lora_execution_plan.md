# TRM LoRA 微调代码改造全景方案

为了在微型递归模型 (TRM) 上实现“无遗憾”的高效 LoRA 微调（基于 `r=64` 的全线性层注入策略），且**绝不破坏原有的预训练代码库（如 `pretrain.py`, `trm.py`）**。我们将采用纯外挂旁路的重构方式。

以下是将要新增或修改的具体文件位置及职责：

---

### 第一步：定制底层神经元 (LoRA 注入点)
**新增文件：** `models/lora_layers.py`

*   **设计原因**：由于 TRM 内部完全使用了自定义的 `CastedLinear` 而非标准 `torch.nn.Linear`，现成的 `peft` 库无法识别和挂载外挂矩阵。
*   **具体实施**：
    1.  声明 `LoRACastedLinear` 类：完美包装原有的 `CastedLinear` 实例，内部新增维度分别为 `[in_dim, r]` 和 `[r, out_dim]` 的 `A`、`B` 可训练参数矩阵。前向传播实现为：$Output = Original(X) + (X \times A \times B) \times \frac{\alpha}{r}$。
    2.  声明替换工厂函数 `inject_lora(model, r=64, alpha=32)`：采用类似深度优先遍历的方法，扫描模型底层的子模块，把所有识别出的 `CastedLinear` **原地无缝替换** 为带有小矩阵的 `LoRACastedLinear`。

### 第二步：编写全新的主控微调脚本
**新增文件：** `lora_finetune.py`

*   **设计原因**：为了与预训练任务物理隔离，我们将从 `pretrain.py` 完整克隆一份代码到该文件，并专门针对微调需求进行两处精准“搭桥”。
*   **具体实施**：
    1.  **打冰封与装外挂**：在模型被 `hydra` 初始化后，追加一行代码将全身所有参数遍历设置 `requires_grad=False`。紧接着调用 `inject_lora()` 挂上新矩阵（只有新矩阵默认带梯度）。
    2.  **吞经验包**：利用系统原有的 `load_checkpoint()` 工具，吃进类似 `checkpoints/ARC-AGI-1/step_155718` 这个 1.8G 的预训练 Checkpoint（此动作必须在注入 LoRA **执行之后**，或者使用 strict=False，确保预先权重被主骨架吸纳，而不管外挂矩阵）。
    3.  **本地文件自归档（纯粹本地方案）**：考虑到服务器环境极其封闭，无法使用任何外联面板（如 WandB 或 SwanLab），我们在启动脚本 `run_lora_arc1.sh` 中启用了最高级别的本地留档方案。
        *   代码使用了 `nohup ... > terminal_output.log 2>&1 &`。
        *   **您的视角**：完全不需要复杂的配置。您每次运行不同超参实验时，模型权重和长卷幅的打印日志（包括进度条、每一轮的 loss）都被死死钉在 `checkpoints/LoRA_Experiments/` 底下对应的实验名文件夹里。同事只需要把这个结果跑完发回给您就好。
    4.  **字符级保底输出**：为了即便不装看管版面也能扫一眼就懂，在 `tqdm` 进度条里附加实时的 `LoRA_Loss` 移动均值打印，并在程序开始处显眼地用打印框打出当前实验的全部超参，作为最后一道保险。

### 第三步：新增一份微调专属超参配置单
**新增文件：** `config/cfg_lora_finetune.yaml`

*   **设计原因**：根据 "LoRA Without Regret" 指引，微调环境的超参需要特殊对待。我们新开配置单防止污染全局。
*   **具体实施**：
    1.  在默认配置之上，将 `lr` 拔高整整 10 倍（例如设定为 `1e-3` 到 `3e-3`）。
    2.  强制彻底关闭平滑移动平均配置：`ema: False`。
    3.  利用 4 张 H200 的强大算力，将全局批处理大小 `global_batch_size` 显著提升至中大区间（如 `128` 到 `512` 之间）。*注：虽然算力充足，但考虑到 TRM 是在 latent 空间进行多次循环展开的递归模型，过大的 Batch Size（如直接拉满到几千）可能引发非预期的交叉梯度震荡。建议从 `128` 或 `256` 开始尝试。*

### 第四步：编写一键点火壳脚本
**新增文件：** `scripts/run_lora_arc1.sh`

*   **设计原因**：为了让用户可以一键在后台多卡上稳定执行实验。
*   **具体实施**：
    1.  写入 `torchrun` 分布式指令，调用刚刚写好的 `lora_finetune.py`。
    2.  默认指定加载 `cfg_lora_finetune.yaml` 配置。
    3.  配置写明挂载用户提供的 Checkpoint 样本：`+load_checkpoint=checkpoints/ARC-AGI-1/step_155718`。
