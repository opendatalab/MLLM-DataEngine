#!/bin/bash

MODEL_PATH=$1
CKPT=$2

python -m llava.eval.model_vqa_loader \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}.json
