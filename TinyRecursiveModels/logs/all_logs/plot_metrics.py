import os
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def parse_log_for_pass2(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        matches = re.findall(r"'ARC/pass@2':\s*([\d.]+)", content)
        if matches:
            return float(matches[-1])
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return None

def main():
    base_dir = "d:/Project with Jiefu/HyperCircuit/TinyRecursiveModels/logs/all_logs"
    
    # 1. LR Data
    lr_dir = os.path.join(base_dir, "2048_lr_test")
    lr_files = glob.glob(os.path.join(lr_dir, "*.log"))
    
    lr_data = {}
    for f in lr_files:
        basename = os.path.basename(f)
        match = re.search(r"LoRA_lr(.+?)_ckpt(\d+)_", basename)
        if match:
            lr_str = match.group(1)
            ckpt = int(match.group(2))
            val = parse_log_for_pass2(f)
            if val is not None:
                if lr_str not in lr_data:
                    lr_data[lr_str] = []
                lr_data[lr_str].append((ckpt, val))
    
    plt.figure(figsize=(10, 6))
    # sort lr_strs for legend consistency
    def parse_lr(lr_str):
        if 'e' in lr_str:
            return float(lr_str)
        return float(lr_str)
    
    sorted_lrs = sorted(lr_data.keys(), key=parse_lr)
    for lr_str in sorted_lrs:
        data = sorted(lr_data[lr_str], key=lambda x: x[0])
        ckpts = [x[0] for x in data]
        vals = [x[1] for x in data]
        plt.plot(ckpts, vals, marker='o', label=f'LR: {lr_str}')
    
    plt.title('Pass@2 vs Checkpoint for different Learning Rates')
    plt.xlabel('Checkpoint')
    plt.ylabel('Pass@2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lr_test_pass2.png')
    print("Saved lr_test_pass2.png")
    
    # 2. Rank Data
    rank_dir = os.path.join(base_dir, "2048_rank_test")
    rank_files = glob.glob(os.path.join(rank_dir, "*.log"))
    
    rank_data = {}
    for f in rank_files:
        basename = os.path.basename(f)
        match = re.search(r"LoRA_r(\d+)_ckpt(\d+)_", basename)
        if match:
            r_str = int(match.group(1))
            ckpt = int(match.group(2))
            val = parse_log_for_pass2(f)
            if val is not None:
                if r_str not in rank_data:
                    rank_data[r_str] = []
                rank_data[r_str].append((ckpt, val))
                
    plt.figure(figsize=(10, 6))
    sorted_ranks = sorted(rank_data.keys())
    for r_str in sorted_ranks:
        data = sorted(rank_data[r_str], key=lambda x: x[0])
        ckpts = [x[0] for x in data]
        vals = [x[1] for x in data]
        plt.plot(ckpts, vals, marker='s', label=f'Rank: {r_str}')
        
    plt.title('Pass@2 vs Checkpoint for different LoRA Ranks')
    plt.xlabel('Checkpoint')
    plt.ylabel('Pass@2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rank_test_pass2.png')
    print("Saved rank_test_pass2.png")

if __name__ == '__main__':
    main()
