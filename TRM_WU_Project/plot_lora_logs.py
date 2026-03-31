import os
import re
import ast
import matplotlib.pyplot as plt

log_dir = r"D:\zjq\HyperCircuit\TinyRecursiveModels\logs\0320logs"
output_dir = r"C:\Users\11152\.gemini\antigravity\brain\f42b49ba-aa08-4be7-9e58-3edcb450ead9"

# We have two checkpoints
ckpts = ["155422", "310843"]
metrics = ['ARC/pass@1', 'ARC/pass@2', 'ARC/pass@5', 'ARC/pass@10', 'ARC/pass@100']

def parse_logs(ckpt):
    results = {}
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'): continue
        if f"ckpt{ckpt}" not in filename: continue
        
        # Extract rank from filename, e.g., LoRA_r8_ckpt155422_0319_1857.log
        match = re.search(r'LoRA_r(\d+)_ckpt', filename)
        if not match: continue
        rank = int(match.group(1))
        
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                final_eval_line = None
                baseline_eval_line = None
                for line in lines:
                    if 'BASELINE EVAL RESULT (pre-LoRA):' in line:
                        baseline_eval_line = line
                    elif 'EVAL RESULT (Epoch 10000):' in line:
                        final_eval_line = line
                
                # If training finished, use final eval. In some cases, maybe only baseline was recorded?
                # Actually, the user asked for pass@k for LORA rank. 
                # It means we need the final eval (Epoch 10000).
                # If not found, skip.
                if final_eval_line:
                    res_str = final_eval_line.split('EVAL RESULT (Epoch 10000): ')[1].strip()
                    # ast.literal_eval doesn't like np.float32(...). 
                    # A quick regex to strip out np.float32(...) wrappers
                    res_str = re.sub(r'np\.float32\((.*?)\)', r'\1', res_str)
                    try:
                        res_dict = ast.literal_eval(res_str)
                        results[rank] = {k: res_dict.get(k, 0) for k in metrics}
                    except Exception as e:
                        print(f"Error parsing {filename}: {e}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return results

# Baseline metrics from eval_0318_1121.log
baselines = {
    "155422": {
        'ARC/pass@1': 0.36875, 'ARC/pass@2': 0.405, 'ARC/pass@5': 0.45, 'ARC/pass@10': 0.4675, 'ARC/pass@100': 0.535
    },
    "310843": {
        'ARC/pass@1': 0.3975, 'ARC/pass@2': 0.41875, 'ARC/pass@5': 0.47375, 'ARC/pass@10': 0.51, 'ARC/pass@100': 0.5575
    }
}

for ckpt in ckpts:
    res = parse_logs(ckpt)
    if not res:
        print(f"No valid logs found for ckpt {ckpt}")
        continue
    
    # Sort ranks
    ranks = sorted(res.keys())
    
    plt.figure(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, m in enumerate(metrics):
        vals = [res[r][m] for r in ranks]
        color = colors[i % len(colors)]
        
        # Plot LoRA line
        line, = plt.plot(ranks, vals, marker='o', label=m, color=color)
        
        # Plot Baseline as dashed horizontal line
        baseline_val = baselines[ckpt][m]
        plt.axhline(y=baseline_val, color=color, linestyle='--', alpha=0.5, label=f'{m} (Base)')
        
    plt.xlabel('LoRA Rank (r)')
    plt.ylabel('pass@K')
    plt.title(f'pass@K vs LoRA Rank for Checkpoint {ckpt}\n(Dashed lines = Base Model Performance)')
    plt.xscale('log', base=2)
    plt.xticks(ranks, [str(r) for r in ranks])
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Put legend outside to avoid cluttering
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f'lora_rank_search_ckpt{ckpt}.png')
    plt.savefig(out_path)
    print(f"Saved plot for {ckpt} to {out_path}")
    plt.close()
