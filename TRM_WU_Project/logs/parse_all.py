import re, os, sys

output = []

# ====== log0411 (full_trm experiments) ======
log_dir = r'd:\Project with Jiefu\HyperCircuit\TRM_WU_Project\logs\log0411'
for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    fpath = os.path.join(log_dir, fname)
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l]
    step_logs = re.findall(r'Step (\d+)/\d+ \| lm_loss=([0-9.]+)', content)
    nan_count = len(re.findall(r'loss=nan', content))
    
    output.append(f'\n=== {fname} ===')
    output.append(f'NaN: {nan_count}, Final train loss: {step_logs[-1][1] if step_logs else "N/A"}')
    
    for l in eval_lines:
        epoch = re.search(r'Epoch (\d+)', l)
        p1 = re.search(r"'ARC/pass@1': ([0-9.]+)", l)
        p2 = re.search(r"'ARC/pass@2': ([0-9.]+)", l)
        p5 = re.search(r"'ARC/pass@5': ([0-9.]+)", l)
        p10 = re.search(r"'ARC/pass@10': ([0-9.]+)", l)
        lm = re.search(r"'lm_loss': np\.float32\(([0-9.]+)\)", l)
        e = epoch.group(1) if epoch else '?'
        output.append(f'  Epoch {e}: pass@1={p1.group(1) if p1 else "?"}, pass@2={p2.group(1) if p2 else "?"}, pass@5={p5.group(1) if p5 else "?"}, pass@10={p10.group(1) if p10 else "?"}, eval_loss={lm.group(1) if lm else "?"}')

# ====== baseline LoRA ckpt155422 rank_logs ======
output.append('\n\n====== BASELINE: LoRA ckpt155422 (rank_logs) ======')
log_dir2 = r'd:\Project with Jiefu\HyperCircuit\TinyRecursiveModels\logs\rank_logs'
for fname in sorted(os.listdir(log_dir2)):
    if not fname.endswith('.log') or '155422' not in fname: continue
    fpath = os.path.join(log_dir2, fname)
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l]
    if eval_lines:
        last = eval_lines[-1]
        p1 = re.search(r"'ARC/pass@1': ([0-9.]+)", last)
        p2 = re.search(r"'ARC/pass@2': ([0-9.]+)", last)
        p5 = re.search(r"'ARC/pass@5': ([0-9.]+)", last)
        output.append(f'  {fname}: pass@1={p1.group(1) if p1 else "?"}, pass@2={p2.group(1) if p2 else "?"}, pass@5={p5.group(1) if p5 else "?"}')

# ====== baseline pretrain ======
output.append('\n====== BASELINE: Pretrain (E2E) ======')
fpath = os.path.join(log_dir2, 'E2E_Pretrain_0319_1856.log')
if os.path.exists(fpath):
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l or 'pass@' in l]
    for l in eval_lines[:5]:
        output.append(f'  {l[:400]}')
else:
    output.append('  File not found')

# ====== baseline LoRA lr_logs ======
output.append('\n====== BASELINE: LoRA lr_logs ckpt155422 ======')
log_dir3 = r'd:\Project with Jiefu\HyperCircuit\TinyRecursiveModels\logs\lr_logs'
for fname in sorted(os.listdir(log_dir3)):
    if not fname.endswith('.log') or '155422' not in fname: continue
    fpath = os.path.join(log_dir3, fname)
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l]
    if eval_lines:
        last = eval_lines[-1]
        p1 = re.search(r"'ARC/pass@1': ([0-9.]+)", last)
        p2 = re.search(r"'ARC/pass@2': ([0-9.]+)", last)
        p5 = re.search(r"'ARC/pass@5': ([0-9.]+)", last)
        output.append(f'  {fname}: pass@1={p1.group(1) if p1 else "?"}, pass@2={p2.group(1) if p2 else "?"}, pass@5={p5.group(1) if p5 else "?"}')

# ====== baseline.log ======
output.append('\n====== BASELINE: baseline.log ======')
fpath = r'd:\Project with Jiefu\HyperCircuit\TinyRecursiveModels\logs\baseline.log'
if os.path.exists(fpath):
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l or 'pass@' in l]
    for l in eval_lines[:5]:
        output.append(f'  {l[:400]}')

outpath = r'd:\Project with Jiefu\HyperCircuit\TRM_WU_Project\logs\full_comparison.txt'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
print(f'Written to {outpath}')
