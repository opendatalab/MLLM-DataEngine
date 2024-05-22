"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import math
import torch
import base64
from PIL import Image
from io import BytesIO

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class OKVQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        img_file = 'COCO_val2014_{:0>12}.jpg'.format(img_id)
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id

class VizWizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] The question is '{question}' Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"
        return image, question, answers

class IconQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_id = data['image_id']
        question = data['question']
        image_path = os.path.join(self.root_path, image_id, 'image.png')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image).half().cuda()
        candidates = '_'.join(data['choices'])
        answer = data['answer']
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, candidates, answer

class GQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["image"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["question"]
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        labels = ann["answer"]

        return image, question, labels

class HMEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no. Answer:"
        labels = ann["label"]

        return image, question, labels

class VSREvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.root_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["caption"]
        question = f'[vqa] Based on the image, is this statement true or false? {question}'
        labels = 'true' if ann["label"] == 1 else 'false'

        return image, question, labels
    
class SEEDEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path, model_type):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.model_type = model_type

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image = data['image']
        question = data['text']
        question_id = data['question_id']
        img_file = os.path.join(self.root_path, image)
        image = Image.open(img_file).convert('RGB')
        image = self.vis_processor(image)
        if 'vicuna' not in self.model_type:
            question = f"[vqa] {question}"
        return image, question, question_id
    
def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False
    
def load_image_from_base64(image):
        return Image.open(BytesIO(base64.b64decode(image)))
    
class MMBenchEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, model_type):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.model_type = model_type

    def __len__(self):
        return len(self.loaded_data)
    
    def get_options(self, options):
        letters = ['A', 'B', 'C', 'D']
        parsed_letters, parsed_options = [], []
        for letter, option in zip(letters, options):
            if is_none(option):
                continue
            else:
                parsed_letters.append(letter)
                parsed_options.append(option)
        return parsed_letters, parsed_options
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image = load_image_from_base64(data['image']).convert('RGB')
        image = self.vis_processor(image)
        question = data['question']
        question_id = data['index']
        hint = data['hint']
        option_a = data["A"]
        option_b = data["B"]
        option_c = data["C"]
        option_d = data["D"]
        
        parsed_letters, parsed_options = self.get_options(options=[option_a, option_b, option_c, option_d])
        
        question = f"[vqa] {question}"
        if not is_none(hint):
            question = hint + '\n' + question + '\n'
        else:
            question = question + '\n'
        for letter, option in zip(parsed_letters, parsed_options):
            question += f'{letter}. {option}\n'
        question += "Answer with the option's letter from the given choices directly."
        
        return image, question, question_id, [parsed_options], [parsed_letters]