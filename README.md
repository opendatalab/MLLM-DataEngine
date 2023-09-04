# MLLM-DataEngine: An Iterative Refinement Approach for MLLM
Zhiyuan Zhao, Linke Ouyang, Bin Wang, Siyuan Huang

**Shanghai Artificial Intellegence Laboratory**

## Introduction

we propose MLLM-
DataEngine, a novel closed-loop system that bridges data
generation, model training, and evaluation.

![overview](https://github.com/opendatalab/MLLM-DataEngine/assets/40555727/d999de9b-fa86-4ad2-94b1-b78f8e047c44)

![Showcase1](https://github.com/JulioZhao97/MLLM-DataEngine/assets/40555727/847b019d-a980-4db1-b6fa-23a70e0f3abf)


## Getting Started

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/JulioZhao97/MLLM-DataEngine.git
cd MLLM-DataEngine
conda env create -f environment.yml
conda activate minigpt4
pip3 install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```


**2. Prepare the pretrained Vicuna weights**

The current version of MLLM-DataEngine is built on the v0 version of Vicuna-13B.
Please refer to our instruction [here](PrepareVicuna.md) 
to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

**3. Prepare the stage1 pretrained MiniGPT-4 checkpoint**

Download the pretrained checkpoints according to the Vicuna model you prepare.

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) 


Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10) at Line 11. 

**4. Data Preparation**

1. download COCO2017 images and annotations from official website [here](https://cocodataset.org/#download).

2. put coco images and annotations under ```data/```

3. download following datasets, uncompress, and put them under ```data/```

| A-OKVQA | CCSBUAlign | GPTVQA | 
| :---: | :---: | :---: |
| [download](https://drive.google.com/file/d/1RE8nyVzXhIG7GMrYyKiQv10vY-gud2TK/view?usp=drive_link) | [download](https://drive.google.com/file/d/1s7kKpRSB0BVveRY2YGN4uhdCmxjWOfJb/view?usp=drive_link) | [download](https://drive.google.com/file/d/1_5EmALJ_UfN19Fi9iBvNbB2U-LBVcEVD/view?usp=drive_link) |

4. finally check the data structure as follows:

```
.
├── A-OKVQA
│   ├── aokvqa_v1p0_train.json
│   └── aokvqa_v1p0_val_classified.json  # with question type assigned by GPT-4
├── cc_sbu_align
│   ├── filter_cap.json
│   └── image
├── COCO2017
│   ├── annotations
│   │    └── ...
│   ├── train2017
│   │    └── ...
│   └── val2017
│        └── ...
└── gptvqa
    ├── DataEngine_round1_data.json
    └── DataEngine_round2_data.json
```

## Install OpenCompass

To evaluate on MMBenchmark, install OpenCompass evaluation environment according to following instructions:

1. Install OpenCompass from ![here](https://opencompass.readthedocs.io/en/latest/get_started.html#installation).

2. Prepare MiniGPT-4 environment according to ![here](https://github.com/open-compass/opencompass/tree/main/configs/multimodal/minigpt_4).

```
cd opencompass/multimodal/models/minigpt_4

git clone https://github.com/Vision-CAIR/MiniGPT-4.git


```

3. 

## Model Training & Evaluation

**1. model tuning**

We add round1 and round2 data of GPTVQA into instruct tuning. After preparing the data and model, setting ```--cfg-path``` and run the following command. In our experiments, we use 4 A100 gpus. We add lora weights to finetune LLM, which are saved in checkpoint.

in train config, set the ```llama_model``` to the path of vicuna model, and set ```ckpt``` to the stage1 pretrained MiniGPT-4 model.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path path/to/config
```

Train configs, finetuned model, and results are shown in following table.

| **LLM** | **Base SFT Data** | **Round1** | **Round2** | **format** | **MMBench dev** | **AOKVQA val (MC)** | **AOKVQA val (DA)** | **config** | **checkpoint** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7B | CCSBUAlign, A-OKVQA |  |  | QMA | 45.8 | 70.2 | 59.1 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMA/minigpt4_7b_stage2_finetune_aokvqa.yaml) | [model](https://drive.google.com/file/d/1rsYx7Uc-fB8-1ZgKTNqIGmQ_tbNzf6vP/view?usp=drive_link) |
| 7B | CCSBUAlign, A-OKVQA | :heavy_check_mark: |  | QMA | 47.1 | 71.8 | 60.8 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMA/minigpt4_7b_stage2_finetune_aokvqa_round1.yaml) | [model](https://drive.google.com/file/d/1qzpUriJK1yKA274nH9DTHPimaepz-EdM/view?usp=drive_link) |
| 7B | CCSBUAlign, A-OKVQA | :heavy_check_mark: | :heavy_check_mark: | QMA | 52.7 | 73.6 | 62.0 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMA/minigpt4_7b_stage2_finetune_aokvqa_round1_round2.yaml) | [model](https://drive.google.com/file/d/19j_FmsIzfGBhH7sOLts51IVff4XgDYbn/view?usp=sharing) |
| 7B | CCSBUAlign, A-OKVQA |  |  | QMAE | 25.7 | 71.0 | 59.1 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMAE/minigpt4_7b_stage2_finetune_aokvqa.yaml) | [model](https://drive.google.com/file/d/1HR8hknybstr--sOX2C7TmWeiXIiiIXCD/view?usp=sharing) |
| 7B | CCSBUAlign, A-OKVQA | :heavy_check_mark: |  | QMAE | 40.6 | 71.0 | 60.3 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMAE/minigpt4_7b_stage2_finetune_aokvqa_round1.yaml) | [model](https://drive.google.com/file/d/12ITEv4DAB-3g7f_4XLBtp4JQPz42RkfO/view?usp=sharing) |
| 7B | CCSBUAlign, A-OKVQA | :heavy_check_mark: | :heavy_check_mark: | QMAE | 46.7 | 72.1 | 61.0 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-7b/QMAE/minigpt4_7b_stage2_finetune_aokvqa_round1_round2.yaml) | [model](https://drive.google.com/file/d/1FsAfSiJNICrJOQsKLwF1Y-48-gTvzPHk/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA |  |  | QMA | 52.6 | 74.8 | 62.6 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMA/minigpt4_7b_stage2_finetune_aokvqa.yaml) | [model](https://drive.google.com/file/d/10h4hl7Xprd8cQxln1Xug82HeDY_35LfY/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA | :heavy_check_mark: |  | QMA | 52.5 | 74.7 | 63.1 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMA/minigpt4_7b_stage2_finetune_aokvqa_round1.yaml) | [model](https://drive.google.com/file/d/1u_acXl8SuaVBf9zR_Rwq3S3c5_svrCyy/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA | :heavy_check_mark: | :heavy_check_mark: | QMA | 56.1 | 75.5 | 63.3 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMA/minigpt4_7b_stage2_finetune_aokvqa_round1_round2.yaml) | [model](https://drive.google.com/file/d/19XQ4HWH4XuOJe3tMYQVYlUgLwFlqdPz_/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA |  |  | QMAE | 46.1 | 73.1 | 62.1 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMAE/minigpt4_7b_stage2_finetune_aokvqa.yaml) | [model](https://drive.google.com/file/d/1qhmQQQB4xz5S0ooCYwjLuanTjy3ugwGP/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA | :heavy_check_mark: |  | QMAE | 48.1 | 74.5 | 62.4 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMAE/minigpt4_7b_stage2_finetune_aokvqa_round1.yaml) | [model](https://drive.google.com/file/d/1_Oo3o7l01nCc23WLmWYHvosPt0Gqai97/view?usp=sharing) |
| 13B | CCSBUAlign, A-OKVQA | :heavy_check_mark: | :heavy_check_mark: | QMAE | 49.2 | 74.0 | 61.9 | [config](https://github.com/JulioZhao97/MLLM-DataEngine/blob/main/train_configs/ccsbualign_aokvqa/vicuna-13b/QMAE/minigpt4_7b_stage2_finetune_aokvqa_round1_round2.yaml) | [model](https://drive.google.com/file/d/11TQ1XRdntvlBmUGVUkJ3b_We183CEtHg/view?usp=sharing) |

**2. merge lora weight into LLM**

after finetuning, run following command to merge lora weight into LLM:

```
python apply_lora_delta.py --base-model path/to/vicuna/weight \
    --ckpt path/to/finetuned/model \
    --target path/to/merged/llm
```

**3. evaluation**

#### evaluate on A-OKVQA

For evaluation on A-OKVQA, run following commands:

```
torchrun --nproc-per-node 4 --master-port $RANDOM train.py --cfg-path eval_configs/minigpt4_eval.yaml
```

in ```eval_configs/minigpt4_eval.yaml```, please change ```llama_model``` to the path of merged LLM, and set ```ckpt``` to the stage1 pretrained MiniGPT-4 model.

Three results files are stored during evaluation:

1. ```evaluation/aokvqa_eval.json```: stores each question and corresponding model answer.

2. ```engine_pipeline/data/bad_case_aokvqa_classified.json```: stores questions which model answered wrongly on each kind of question type (bad cases).

3. ```engine_pipeline/data/weight.json```: model error rate on each kind of question type.

These files are used in follow-up data-engine pipeline.

#### evaluate on MMBenchmark dev

according to limits of MMBenchmark, we cannot release the inference code on MMBenchmark, so we give a brief instruction instead.

1. go to [here](https://github.com/InternLM/opencompass/blob/mm/docs/en/MMBench.md), download MMBenchmark dev data and write inference code according to instructions.

2. during inference, set ```llama_model``` to merged LLM and run inference on MMBenchmark dev, we give config of ```llama_model.generate``` for conveniece:

```
max_new_tokens=10
num_beams=5
do_sample=False
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=-1.0
temperature=1.0
```

## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) This reposity utilize MiniGPT-4 as codebase and base model.
+ [Lavis](https://github.com/salesforce/LAVIS) Fantastic vision-language model codebase.


If you're using MiniGPT-4 in your research or applications, please cite using this BibTeX:
```bibtex
@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
```


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).
