import os
import re
import ast
import matplotlib.pyplot as plt

log_dir = r"D:\zjq\HyperCircuit\TinyRecursiveModels\logs\lora_logs"

# We test the checkpoints that we have baselines for
ckpts = ["155422", "310843"]
metrics = ['ARC/pass@1', 'ARC/pass@2', 'ARC/pass@5', 'ARC/pass@10', 'ARC/pass@100']

# Baseline metrics from eval_0318_1121.log
baselines = {
    "155422": {
        'ARC/pass@1': 0.36875, 'ARC/pass@2': 0.405, 'ARC/pass@5': 0.45, 'ARC/pass@10': 0.4675, 'ARC/pass@100': 0.535
    },
    "310843": {
        'ARC/pass@1': 0.3975, 'ARC/pass@2': 0.41875, 'ARC/pass@5': 0.47375, 'ARC/pass@10': 0.51, 'ARC/pass@100': 0.5575
    }
}

def parse_logs(ckpt):
    results = {}
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'): continue
        if f"ckpt{ckpt}" not in filename: continue
        
        # Extract LR from filename, e.g., LoRA_lr5e-3_ckpt310843_0320_1921.log
        match = re.search(r'LoRA_lr(.*?)_ckpt', filename)
        if not match: continue
        lr_str = match.group(1)
        try:
            lr_val = float(lr_str)
        except ValueError:
            continue
        
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                final_eval_line = None
                for line in lines:
                    if 'EVAL RESULT (Epoch 10000):' in line:
                        final_eval_line = line
                
                if final_eval_line:
                    res_str = final_eval_line.split('EVAL RESULT (Epoch 10000): ')[1].strip()
                    res_str = re.sub(r'np\.float32\((.*?)\)', r'\1', res_str)
                    try:
                        res_dict = ast.literal_eval(res_str)
                        results[lr_val] = {'str': lr_str, 'metrics': {k: res_dict.get(k, 0) for k in metrics}}
                    except Exception as e:
                        print(f"Error parsing {filename}: {e}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return results

for ckpt in ckpts:
    res = parse_logs(ckpt)
    if not res:
        print(f"No valid logs found for ckpt {ckpt}")
        continue
    
    # Sort LRs ascending
    lrs_sorted = sorted(res.keys())
    lr_strs = [res[lr]['str'] for lr in lrs_sorted]
    
    plt.figure(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, m in enumerate(metrics):
        vals = [res[lr]['metrics'][m] for lr in lrs_sorted]
        color = colors[i % len(colors)]
        
        # Plot LoRA line
        line, = plt.plot(lrs_sorted, vals, marker='o', label=m, color=color)
        
        # Plot Baseline as dashed horizontal line
        baseline_val = baselines[ckpt][m]
        plt.axhline(y=baseline_val, color=color, linestyle='--', alpha=0.5, label=f'{m} (Base)')
        
    plt.xlabel('Learning Rate (lr)')
    plt.ylabel('pass@K')
    plt.title(f'pass@K vs Learning Rate for Checkpoint {ckpt}\n(Dashed lines = Base Model Performance)')
    plt.xscale('log')
    plt.xticks(lrs_sorted, lr_strs)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Put legend outside to avoid cluttering
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = os.path.join(log_dir, f'lora_lr_search_ckpt{ckpt}.png')
    plt.savefig(out_path)
    print(f"Saved plot for {ckpt} to {out_path}")
    plt.close()
