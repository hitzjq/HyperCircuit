import sys
import subprocess
import argparse

import yaml

template = "python scripts/train_custom_sft.py configs/lora_gsm8k.yaml --exp_setup=oracle_lora --train_ds_names={task} --eval_ds_info='{{\"{eval_task}\": {{}}}}' --model_dir={model_dir} --lr=8e-5 --batch_size=8 --grad_accum_steps=2 --val_batch_size=32 --epochs=100 --save_to_base_model_dir=True --val_freq=500"


def run_template(model_dir, tasks):
    if isinstance(tasks, str):
        tasks = [tasks]
    for task in tasks:
        eval_task = task
        if task == "magicoder":
            eval_task = "humaneval"
        command = template.format(model_dir=model_dir, task=task, eval_task=eval_task)
        if "Phi-3" in model_dir:
            command += " --target_modules=qkv_proj"

        subprocess.run(command, shell=True)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-config", type=str, required=True)
    # args = parser.parse_known_args()[0]
    # conf = yaml.safe_load(open(args.config))
    model_dir = sys.argv[1]
    tasks = sys.argv[2:]
    run_template(model_dir, tasks)
