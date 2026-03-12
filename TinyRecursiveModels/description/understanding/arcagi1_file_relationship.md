# ARC-AGI-1 预训练文件调用关系图

当您在终端输入 `bash run_ARCAGI1.sh` 并按下回车时，整个项目的代码文件就像一个齿轮组一样开始连锁运转。以下流向图为您直观且严谨地展现了这段代码背后的“谁调用了谁”：

```mermaid
graph TD
    %% 阶段一：外部启动层
    subgraph "🚀 1. 外部启动命令"
        A[run_ARCAGI1.sh]:::bash_script
        A -- "执行 torchrun 调用" --> B1(pretrain.py)
        A -- "覆写参数 L_layers=2 等" --> C(config/cfg_pretrain.yaml)
    end

    %% 阶段二：配置与数据准备层
    subgraph "⚙️ 2. 初始化与供料层"
        C -. "提供默认全局配置" .-> B1
        B1 -- "读取路径实例化 DataSet" --> D1[puzzle_dataset.py]
        
        D1 -- "从硬盘加载海量 .npy" --> D2[(data/arc1concept-aug-1000/)]
        D2 -. "返回 inputs/labels/identifiers Tensors" .-> D1
        D1 -. "把组装好的 Batch 喂给模型" .-> B1
    end

    %% 阶段三：模型骨架实例化
    subgraph "🧠 3. 模型核心架构实例化"
        B1 -- "根据 config 实例化模型类" --> E1[models/recursive_reasoning/trm.py]
        
        E1 -- "构建 ACT 递归执行主体" --> E2[models/layers.py]
        E1 -- "构建巨大题号查表矩阵" --> E3[models/sparse_embedding.py]
        
        E2 -. "提供 CastedLinear/SwiGLU 物理零件" .-> E1
        E3 -. "提供 Puzzle Embedding 特征" .-> E1
    end

    %% 阶段四：前向传播与算分
    subgraph "⚖️ 4. 计算与反向传播大循环"
        B1 -- "每个 batch 传给模型前向推演" --> F1{前向传播 (Forward)}
        F1 -- "使用 TRM 算出预测和 Halt 决策" --> E1
        
        E1 -- "拿出结果算交叉熵" --> F2[models/losses.py]
        F2 -. "返回梯度 Loss 用于更新权重" .-> B1
        
        B1 -- "阶段性期末考试时调用" --> G[evaluators/arc.py]
        G -. "返回 Exact Match Score" .-> B1
    end

    %% 样式设定
    classDef bash_script fill:#f9f,stroke:#333,stroke-width:2px;
    classDef file fill:#bbf,stroke:#333,stroke-width:1px;
    classDef data fill:#dfd,stroke:#333,stroke-width:1px;

    class B1,C,D1,E1,E2,E3,F2,G file;
    class D2 data;

```

### 图解核心路径（以一个 Batch 的寿命为例）：

1. **起点**：您执行 `run_ARCAGI1.sh`。它带着您改写的参数配置去调用 `pretrain.py`。
2. **取水**：`pretrain.py` 开始执行 `for batch in dataloader:`，这会激活 `puzzle_dataset.py`。`puzzle_dataset.py` 跑到生成好的 `data` 文件夹里，挖出一批 `labels` 和那堆数字号码牌 `puzzle_identifiers`。
3. **推演**：`pretrain.py` 把这批数据一股脑塞进实例化的 `trm.py` 长长的管子里。
    * 先过 `sparse_embedding.py`，根据题号牌查出专属记忆向量，拼装在序列最前面。
    * 然后带着这个专属记忆，进入由 `layers.py` (包含 CastedLinear) 提供零件组成的递归大循环 `L_cycles`。
4. **交卷**：`trm.py` 吐出预测出的颜色像素块。
5. **批卷**：`pretrain.py` 把预测结果扔给 `losses.py` 去对照参考答案计算得分。算出来的极小数值顺着刚才这根错综复杂的管子退回去，稍微扭转一下每一层的参数。
6. **期末结课**：等几千个 Batch 跑完一定轮数（Epoch），`pretrain.py` 短暂停下脚步，请来地狱判官 `evaluators/arc.py` 给目前所有的测试数据跑一个严苛的 Exact Match 准确率。
