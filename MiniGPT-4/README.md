# MLLM-DataEngine for MiniGPT4-v2

## Installation

**1. Prepare environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
```

**2. Prepare the pretrained LLM weights**

**MiniGPT-v2** is based on [Llama2-chat-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
Download the corresponding LLM weights from the following huggingface space via huggingface download.

**3. Prepare the pretrained model checkpoints**

Download the stage-2 pretrained MiniGPT4-v2 checkpoints from [here](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view?usp=sharing) and put it to ```MLLM-DataEngine-v2/MiniGPT-4/checkpoint_stage2.pth```

## Data Preparation

### Download the dataset for finetuning the MiniGPT-v2

Download the dataset

Image source | Download path
--- | :---:
COCO 2014 images | <a href="http://images.cocodataset.org/zips/train2014.zip">images</a> &nbsp;&nbsp;  <a href="https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json"> captions</a>
COCO VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_train.json">vqa train</a> &nbsp;&nbsp;  <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val.json"> vqa val</a>
Visual Genome |  <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip">images part1</a> &nbsp;&nbsp; <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip">images part2</a> &nbsp;&nbsp; <a href="https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"> image meta data </a>
TextCaps | <a href="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip">images</a>  &nbsp;&nbsp; <a href="https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json"> annotations</a> 
RefCOCO | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"> annotations </a>
RefCOCO+ | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"> annotations </a>
RefCOCOg | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"> annotations </a>
OKVQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/okvqa/okvqa_train.json"> annotations </a>
AOK-VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/aokvqa/aokvqa_v1p0_train.json"> annotations </a>
OCR-VQA | <a href="https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing"> annotations </a>
GQA | <a href="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip">images</a>  &nbsp;&nbsp; <a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/gqa/train_balanced_questions.json"> annotations </a>
Filtered flickr-30k |  <a href="https://drive.google.com/drive/folders/19c_ggBI77AvdtYlPbuI0ZpnPz73T5teX?usp=sharing"> annotations </a>
Multi-task conversation |  <a href="https://drive.google.com/file/d/11HHqB2c29hbSk-WLxdta-nG8UCUrcCN1/view?usp=sharing"> annotations </a> 
Filtered unnatural instruction |  <a href="https://drive.google.com/file/d/1lXNnBcb5WU-sc8Fe2T2N8J0NRw4sBLev/view?usp=sharing"> annotations </a>
LLaVA | <a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json"> Compelex reasoning </a> &nbsp;&nbsp;<a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json"> Detailed description </a> &nbsp;&nbsp; <a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json"> Conversation </a> 

### MLLM-DataEngine generated data

Download MLLM-DataEngine generated data from [huggingface](https://huggingface.co/datasets/juliozhao/dataengine_minigpt4) or [opendatalab](https://openxlab.org.cn/datasets/zzy8782180/DataEngine-InstData), and put ```dataengine_minigpt4.json``` under:

```bash
train_dataset
└── data_engine
    └── dataengine_minigpt4.json
...
```

### COCO captions
Download the COCO 2014 images and captions, put them as follows:

```bash
train_dataset
└── COCO2014
    ├── train
    └── coco_karpathy_train.json
...
```

### COCO VQA
Download the vqav2 train and validation json files

```bash
├── train_dataset
│   ├── vqav2
│       ├── vqa_train.json
|       ├── vqa_val.json
```


### Visual genome
Download visiual genome images and annotation files

```bash
train_dataset
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
│   ├── region_descriptions.json
│   └── image_data.json
...
```


### TextCaps
Download the TextCaps images and annotation files

```bash
├── train_dataset
│   ├── textcaps
│       ├── train_images
│       ├── TextCaps_0.1_train.json
```


### RefCOCO, RefCOCO+, RefCOCOg
Download the RefCOCO, RefCOCO+, RefCOCOg annotation files

```bash
train_dataset
├── refcoco
│   ├── refcoco
│   │   ├── instances.json
│   │   ├── refs(google).p
│   │   └── refs(unc).p
│   ├── refcoco+
│   │   ├── instances.json
│   │   └── refs(unc).p
│   └── refcocog
│       ├── instances.json
│       ├── refs(google).p
│       └─── refs(und).p
...
```


### OKVQA

```bash
train_dataset
├── okvqa
    ├── okvqa_train.json
```


### AOK-VQA
Download the AOK-VQA annotation dataset

```
export AOKVQA_DIR=YOUR_DATASET_PATH
mkdir -p ${AOKVQA_DIR}
curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

```bash
train_dataset
├── aokvqa
    ├── aokvqa_v1p0_train.json
```



### OCR-VQA
Download the OCR-VQA annotation files
download the images with loadDataset.py script

```bash
train_dataset
├── ocrvqa
    ├── images
    ├── dataset.json
```


### GQA
Download the GQA annotation files and images

```bash
train_dataset
├── gqa
    ├── images
    ├── train_balanced_questions.json
```


### filtered Flickr-30k
Download filtered Flickr-30k images (fill this [form](https://forms.illinois.edu/sec/229675) on official website or from [kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/download?datasetVersionNumber=1)) and annotation files

```bash
train_dataset
├── filtered_flickr
│   ├── images
│   ├── captiontobbox.json
│   ├── groundedcaption.json
│   └── phrasetobbox.json
...
```


### Multi-task conversation
Download the multi-task converstation dataset

```bash
train_dataset
├── multitask_conversation
│   └── multitask_conversation.json
...
```

### Unnatural instruction
Download the filtered unnatural instruction annotation files (we remove the very long sentences from the original unnatural instruction dataset)

```bash
train_dataset
    ├── unnatural_instructions
        ├── filtered_unnatural_instruction.json
```

### LLaVA

```bash
train_dataset
    ├── llava
        ├── conversation_58k.json
        ├── detail_23k.json
        ├── complex_reasoning_77k.json
```

## Training

We perform the stage-3 training on 8xA100 gpus, which takes 8-10 hours. Run the following command to train model:

```bash
torchrun --master-port $RANDOM --nproc_per_node 8 train.py --cfg-path train_configs/minigptv2_finetune_dataengine.yaml
```

## Evaluation

1. For evaluation on downstream datasets, first download [evaluation dataset](https://drive.google.com/file/d/1j73sLdSbVcztzw5CSzRl6e7mj9ndEsbv/view?usp=drive_link) and put folder under ```MLLM-DataEngine-v2/MiniGPT-4```.

2. Change ```ckpt``` key in ```eval_configs/minigptv2_benchmark_evaluation.yaml``` to the model you trained. Change ```ckpt``` to ```dataengine_minigpt4v2.pth``` if you want to reproduce results in paper, download model from [here](https://huggingface.co/juliozhao/dataengine_minigpt4v2_model/tree/main). 

### SEED-Bench

1. download SEED-Bench images (not video frames) and put under ```evaluation_dataset/SEED-Bench-image```

2. inference on SEED-Bench

```bash
torchrun --master-port $RANDOM --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml --dataset seed
```

3. calculate results

```bash
python eval_scripts/convert_seed_for_submission_minigpt4.py \
    --annotation-file ./evaluation_dataset/seed/SEED-Bench-image.json \
    --result-file ./evaluation_results/seed.jsonl
```

### MMBench

1. Inference on MMBench

```bash
torchrun --master-port $RANDOM --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml --dataset mmbench
```

2. Convert results to MMBench format

```bash
python eval_scripts/convert_mmbench_for_submission.py \
    --annotation-file evaluation_dataset/mmbench/mmbench_dev_20230712.tsv \
    --result-file evaluation_results/mmbench.jsonl \
    --output-file evaluation_results/mmbench.xlsx
```

3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission)

### OKVQA, VizWiz, VSR

**COCO2014 val**: download COCO2014 validation images and put under ```evaluation_dataset/coco2014_val/```

**VizWiz**: download vizwiz validation set images from [here](https://vizwiz.org/tasks-and-datasets/vqa/) and put under ```evaluation_dataset/vizwiz/vizwiz_images```

**VSR**: download VSR images from [here](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) and put under ```evaluation_dataset/vsr/vsr_images```

```bash
torchrun --master-port $RANDOM --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path ./eval_configs/minigptv2_benchmark_evaluation.yaml --dataset okvqa,vizwiz,vsr
```

### Main Results

| Incremental Dataset | Data Amount | SEED | MMB | OKVQA | VizWiz | VSR |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| None(baseline) | - | 49.21 | 38.83 | 56.03 | 53.08 | 61.37 |
| MLLM-DataEngine | 270k | **63.83** | **52.92** | **56.87** | **54.39** | **62.43** |