#!/bin/bash

uv run python scripts/run_eval.py --model-dir  mistralai/Mistral-7B-Instruct-v0.2 --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
uv run python scripts/run_eval.py --model-dir  meta-llama/Llama-3.1-8B-Instruct --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp
uv run python scripts/run_eval.py --model-dir  google/gemma-2-2b-it --use-icl --save-to-base-model-dir --tasks boolq winogrande piqa hellaswag arc_easy arc_challenge openbookqa gsm8k humaneval mbpp