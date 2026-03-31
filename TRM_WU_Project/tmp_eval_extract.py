import sys

files = [
    r"d:\zjq\HyperCircuit\TinyRecursiveModels\logs\LoRA_r64_ckpt259_unfreezeTrue_0318_1134.log",
    r"d:\zjq\HyperCircuit\TinyRecursiveModels\logs\LoRA_r64_ckpt310_unfreezeTrue_0318_1134.log"
]

for f in files:
    print(f"\n--- {f} ---")
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
            for line in fp:
                if 'EVAL RESULT' in line or 'EVALUATION SUMMARY' in line:
                    # Strip emoji or encode properly to cp1252/gbk safely
                    print(line.encode('ascii', 'ignore').decode('ascii').strip())
    except Exception as e:
        print(f"Error: {e}")
