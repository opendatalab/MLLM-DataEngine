import os
import json
import PIL
import logging
from PIL import Image
from PIL import ImageFile
from statistics import mode
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from io import BytesIO
import random
import numpy as np

class AOKVQADataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, anno_path, coco_anno, format="Q-A", sample_num=None):
        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # read annotation from ceph
        if self.anno_path.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.samples = json.loads(client.get(self.anno_path))
        else:
            self.samples = json.load(open(self.anno_path, 'r'))
        
        # empty answer
        valid_id = []
        for i, sample in enumerate(self.samples):
            if len(sample["choices"][sample["correct_choice_idx"]]) == 0:
                continue
            valid_id.append(i)
        self.samples = [self.samples[i] for i in valid_id]
        
        if sample_num is not None:
            logging.info(f"random sample {sample_num} data")
            random.shuffle(self.samples)
            self.samples = self.samples[:sample_num]
        
        # QA format
        self.format = format
        
        self.coco_anno = json.load(open(coco_anno, "r"))
        self.id2file = {img["id"]:img["file_name"] for img in self.coco_anno["images"]}
            
        if vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            self.reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            self.reader = {'type': 'LocalReader', 'body': Image.open}
        
        # filter for errors in post-processing
        valid_idx = []
        for idx, sample in enumerate(self.samples):
            if "" in sample["choices"]:
                continue
            valid_idx.append(idx)
        self.samples = [self.samples[idx] for idx in valid_idx]
        
        # filter "description" in explaination
        valid_idx = []
        for idx, sample in enumerate(self.samples):
            if "description" in sample["rationales"][0] or "descriptions" in sample["rationales"][0]:
                continue
            valid_idx.append(idx)
        self.samples = [self.samples[idx] for idx in valid_idx]
        
        valid_idx = []
        for idx, sample in enumerate(self.samples):
            if isinstance(sample["image_id"], str):
                continue
            valid_idx.append(idx)
        self.samples = [self.samples[idx] for idx in valid_idx]
        
        print('total {} vqa samples'.format(self.__len__()))
        
    def __len__(self):
        return len(self.samples)

    def make_choice_text(self, choices, correct_choice_idx):
        letters = "ABCDEFG"
        choice_txt = ""
        
        for i in range(len(choices)):
            choice_txt += f"({letters[i]}) {choices[i]} "
        correct_letter = f"({letters[correct_choice_idx]})"
        return choice_txt.strip(), correct_letter
    
    def make_mm_choice_text(self, choices, correct_choice_idx):
        letters = "ABCDEFG"
        choice_txt = ""
        for i in range(len(choices)):
            choice_txt += f"{letters[i]}. {choices[i]}\n"
        correct_letter = f"({letters[correct_choice_idx]})"
        return choice_txt.strip(), correct_letter
    
    def __getitem__(self, index):
        ann = self.samples[index]
        
        choices = ann["choices"]
        correct_choice_idx = ann["correct_choice_idx"]
        
        question, answer = ann['question'], ann["choices"][correct_choice_idx]
        answer, question = answer.strip(), question.strip()
        
        rationales = sorted(ann["rationales"], key=lambda x:len(x), reverse=True)
        rational = rationales[0]
        
        if self.format == "Q-A":
            mask_text = f"{question}\n###Assistant: "
            qa_text = f"{question}\n###Assistant: The answer is {answer}."
        elif self.format == "QM-A":
            choice_txt, correct_letter = self.make_choice_text(choices, correct_choice_idx)
            mask_text = f"{question}\n{choice_txt}\n###Assistant: "
            qa_text = f"{question}\n{choice_txt}\n###Assistant: The answer is {correct_letter} {answer}."
        elif self.format == "QM-AE":
            choice_txt, correct_letter = self.make_choice_text(choices, correct_choice_idx)
            mask_text = f"{question}\n{choice_txt}\n###Assistant: "
            qa_text = f"{question}\n{choice_txt}\n###Assistant: The answer is {correct_letter} {answer}. {rational}"
        elif self.format == "mmbench":
            choice_txt, correct_letter = self.make_mm_choice_text(choices, correct_choice_idx)
            mask_text = f"{question}\n{choice_txt}\n###Assistant: "
            qa_text = f"{question}\n{choice_txt}\n###Assistant: The answer is {correct_letter} {answer}. {rational}"
        
        image_id = ann['image_id']
        image_path = os.path.join(self.vis_root, self.id2file[image_id])
        
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        image = self.vis_processor(image)
        
        return {
            "image": image,
            "mask_text": mask_text,
            "text_input": qa_text,
            "data_type": "multi_choice_vqa",
        }