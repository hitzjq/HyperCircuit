"""
统一管理 Circuit 提取管线的 run 命名和路径。

使用方式:
    from run_config import RunConfig
    rc = RunConfig(run_name="auto")  # 自动生成带时间的名字
    rc = RunConfig(run_name="arc1_step5000_0415_1617")  # 手动指定

目录结构:
    CodeCircuit_TRM_Arc1/runs/<run_name>/
        config.json              ← 记录参数
        activations/
            block_0/             ← Step 1 输出
            block_1/
        checkpoints/
            sae_block_0.pt       ← Step 2 输出
            sae_block_1.pt
            trm_cross_layer_transcoder/  ← CLT 格式
        attribution_graphs/
            graph_000000.pt      ← Step 3 输出
            ...
        cc_advanced_features.pt  ← Step 4 最终输出
        logs/                    ← 日志
"""

import os
import json
from datetime import datetime


BASE_DIR = "CodeCircuit_TRM_Arc1/runs"


class RunConfig:
    """Circuit 提取管线的 run 路径管理器"""
    
    def __init__(self, run_name="auto", base_dir=BASE_DIR):
        if run_name == "auto":
            run_name = f"run_{datetime.now().strftime('%m%d_%H%M')}"
        
        self.run_name = run_name
        self.run_dir = os.path.join(base_dir, run_name)
        
        # 4 步的输入/输出路径
        self.activations_dir = os.path.join(self.run_dir, "activations")
        self.block_0_dir = os.path.join(self.activations_dir, "block_0")
        self.block_1_dir = os.path.join(self.activations_dir, "block_1")
        
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.sae_block_0_path = os.path.join(self.checkpoints_dir, "sae_block_0.pt")
        self.sae_block_1_path = os.path.join(self.checkpoints_dir, "sae_block_1.pt")
        self.clt_dir = os.path.join(self.checkpoints_dir, "trm_cross_layer_transcoder")
        
        self.graphs_dir = os.path.join(self.run_dir, "attribution_graphs")
        
        self.features_path = os.path.join(self.run_dir, "cc_advanced_features.pt")
        
        self.logs_dir = os.path.join(self.run_dir, "logs")
    
    def create_dirs(self):
        """创建所有子目录"""
        for d in [self.block_0_dir, self.block_1_dir, 
                  self.checkpoints_dir, self.graphs_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
    
    def save_config(self, extra_info=None):
        """保存本次 run 的参数到 config.json"""
        config = {
            "run_name": self.run_name,
            "created_at": datetime.now().isoformat(),
            "paths": {
                "activations": self.activations_dir,
                "sae_block_0": self.sae_block_0_path,
                "sae_block_1": self.sae_block_1_path,
                "graphs": self.graphs_dir,
                "features": self.features_path,
            }
        }
        if extra_info:
            config["params"] = extra_info
        
        os.makedirs(self.run_dir, exist_ok=True)
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"📝 Run config saved: {config_path}")
    
    def print_summary(self):
        """打印本次 run 的路径摘要"""
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  Run: {self.run_name}")
        print(f"  Dir: {self.run_dir}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    @staticmethod
    def add_run_args(parser):
        """给 argparse 添加统一的 --run_name 参数"""
        parser.add_argument(
            "--run_name", type=str, default="auto",
            help="Run 名称。'auto' 自动生成带时间的名字 (run_MMDD_HHMM)。"
        )
        return parser
