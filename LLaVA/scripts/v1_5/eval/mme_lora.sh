#!/bin/bash

MODEL_PATH=$1
CKPT=$2

python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}

python eval_tool/calculation.py --results_dir eval_tool/answers/${CKPT}
