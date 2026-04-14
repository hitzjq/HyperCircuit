import re, os, sys

output = []
log_dir = r'd:\Project with Jiefu\HyperCircuit\TRM_WU_Project\logs\logs0413'

for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    fpath = os.path.join(log_dir, fname)
    fsize = os.path.getsize(fpath)
    
    with open(fpath, 'r', encoding='windows-1252', errors='ignore') as f:
        content = f.read()
    
    output.append(f'\n=== {fname} ({fsize} bytes) ===')
    
    # Check for errors/crashes
    errors = [l for l in content.split('\n') if 'Error' in l or 'error' in l or 'Traceback' in l or 'CUDA' in l or 'OOM' in l or 'RuntimeError' in l]
    if errors:
        output.append(f'  ERRORS FOUND ({len(errors)} lines):')
        for e in errors[:5]:
            output.append(f'    {e[:200]}')
    
    # NaN count
    nan_count = content.count('loss=nan')
    if nan_count > 0:
        output.append(f'  NaN count: {nan_count}')
    
    # Step logs
    step_logs = re.findall(r'Step (\d+)/(\d+) \| lm_loss=([0-9.]+) \| acc=([0-9.]+) \| exact_acc=([0-9.]+)', content)
    if step_logs:
        output.append(f'  Total steps target: {step_logs[0][1]}')
        output.append(f'  First: Step {step_logs[0][0]}, lm_loss={step_logs[0][2]}, acc={step_logs[0][3]}')
        output.append(f'  Last:  Step {step_logs[-1][0]}, lm_loss={step_logs[-1][2]}, acc={step_logs[-1][3]}')
    else:
        output.append(f'  No training step logs found!')
    
    # Eval results
    eval_lines = [l for l in content.split('\n') if 'EVAL RESULT' in l]
    if eval_lines:
        output.append(f'  Eval count: {len(eval_lines)}')
        for l in eval_lines:
            epoch = re.search(r'Epoch (\d+)', l)
            p1 = re.search(r"'ARC/pass@1': ([0-9.]+)", l)
            p2 = re.search(r"'ARC/pass@2': ([0-9.]+)", l)
            p5 = re.search(r"'ARC/pass@5': ([0-9.]+)", l)
            lm = re.search(r"'lm_loss': np\.float32\(([0-9.]+)\)", l)
            e = epoch.group(1) if epoch else '?'
            output.append(f'    Epoch {e}: p@1={p1.group(1) if p1 else "?"}, p@2={p2.group(1) if p2 else "?"}, p@5={p5.group(1) if p5 else "?"}, eval_loss={lm.group(1) if lm else "?"}')
    else:
        output.append(f'  No eval results!')

    # Check last 20 lines for crash info
    last_lines = content.strip().split('\n')[-20:]
    crash_indicators = [l for l in last_lines if 'Error' in l or 'Killed' in l or 'OOM' in l or 'NCCL' in l]
    if crash_indicators:
        output.append(f'  CRASH INDICATORS in last lines:')
        for c in crash_indicators:
            output.append(f'    {c[:200]}')

outpath = r'd:\Project with Jiefu\HyperCircuit\TRM_WU_Project\logs\logs0413_analysis.txt'
with open(outpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
print(f'Written to {outpath}')
