# MLLM-DataEngine for LLaVA-1.5

## Install

1. Navigate to LLaVA folder
```bash
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Visual Instruction Tuning

1. Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Prepare MLLM-DataEngine generated data

Download MLLM-DataEngine generated data from [huggingface](https://huggingface.co/datasets/juliozhao/dataengine_llava) or [opendatalab](https://openxlab.org.cn/datasets/zzy8782180/DataEngine-InstData), and put the ```dataengine_llava.json``` under ```./playground/data/data_engine```. Next, using the jupyter notebook ```./playground/data/process_engine_data.ipynb``` to convert the data format into LLaVA format.

3. Start training!

We use LLaVA-v1.5-7b-lora during experiment, which is based on [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) model during visual instruct tuning, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. No action is needed. 

Run the following command to fine-tune with MLLM-DataEngine generated data. Visual instruct tuning takes about 20h on 8xA100 gpus.

```bash scrips/v1_5/finetune_lora_dataengine.sh```

## Evaluation

### Data Preparation

Before preparing task-specific data, you **MUST** first download [eval.tar.gz](https://drive.google.com/file/d/1Ch-gWG-w_RxXHk_njN-Gg2JtTg59ri4s/view?usp=drive_link). It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to ```./playground/data/eval```. This also provides a general structure for all datasets.

#### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) of **SEED-Bench-1** to download the images. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
3. Multiple-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/seed_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/seed_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```


#### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```
3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/...`.


#### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. put the official image folders under `./playground/data/eval/MME/MME_Benchmark_release_version` as follows:
```bash
./playground/data/eval/MME
├── eval_tool
│   ├── answers
│   └── calculation.py
└── MME_Benchmark_release_version
    ├── artwork
    ├── celebrity
    ├── ...
    └── text_translation
```

4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```

#### GQA

1. Download the [images](https://cs.stanford.edu/people/dorarad/gqa/download.html) and put images under `./playground/data/eval/gqa/data/images`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```

### ScienceQA

1. Download test images from the ScienceQA [repo](https://github.com/lupantech/ScienceQA) and put test images under `./playground/data/eval/scienceqa/images/test`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```

#### VQAv2

1. Download [COCO test2015](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2_lora.sh $MODEL_PATH $MODEL_NAME
# For reproduct results
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2_lora.sh juliozhao/dataenginev2-llava-v1.5-7b-lora dataenginev2-llava-v1.5-7b-lora
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.


### Main Results

| Incremental Dataset | Data Amount | SEED | MMBench | MME | GQA | VQAv2 | ScienceQA |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| None(baseline) | - | 66.04 | 66.66 | 1475/290(1765) | 57.27 | 77.56 | 70.67/68.27 |
| MLLM-DataEngine | 220k | **68.57** | **67.18** | **1511/303(1814)** | **58.02** | **78.18** | **73.17/71.15** |