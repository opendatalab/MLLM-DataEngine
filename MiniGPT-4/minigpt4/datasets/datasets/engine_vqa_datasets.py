"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pdb
import json
import tqdm
import random

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class ENGINEDAVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
        ]

        # load VG annotation
        exist_annotation = []
        for ann in tqdm.tqdm(self.annotation):
            image_path = os.path.join("./train_dataset", ann["image"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vg_path, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["instruction"])
        answer = self.text_processor(ann["answer"])
        question_id = f"engine_da_{index}"

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }
    
    
class ENGINEMCVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
        ]

        # load VG annotation
        exist_annotation = []
        for ann in tqdm.tqdm(self.annotation):
            image_path = os.path.join("./train_dataset", ann["image"])
            if os.path.exists(image_path) and "choice_answer" in ann:
                exist_annotation.append(ann)
        
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vg_path, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = ann["instruction"] + '\n'
        for i, option in enumerate(ann["options"]):
            letter = ["A","B","C","D"][i]
            question += f"{letter}. {option}\n"
        question += "Answer with the option's letter from the given choices directly."
        answer = ann["choice_answer"]
        question_id = f"engine_mc_{index}"

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }
    
    
class ENGINEMCPVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
        ]

        # load VG annotation
        exist_annotation = []
        for ann in tqdm.tqdm(self.annotation):
            image_path = os.path.join("./train_dataset", ann["image"])
            if os.path.exists(image_path) and "choice_answer" in ann:
                # random MC shuffle
                letter_answer = ann["choice_answer"]
                answer_idx = ["A","B","C","D"].index(letter_answer)
                choice_answer_word = ann["options"][answer_idx]
                shuffled_options = [o for o in ann["options"]]
                random.shuffle(shuffled_options)
                shuffled_letter_answer = ["A","B","C","D"][shuffled_options.index(choice_answer_word)]
                shuffled_ann = {k:v for k,v in ann.items()}
                shuffled_ann["options"] = shuffled_options
                shuffled_ann["choice_answer"] = shuffled_letter_answer
                exist_annotation.append(shuffled_ann)
        
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vg_path, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = ann["instruction"] + '\n'
        for i, option in enumerate(ann["options"]):
            letter = ["A","B","C","D"][i]
            question += f"{letter}. {option}\n"
        question += "Answer with the option's letter from the given choices directly."
        answer = ann["choice_answer"]
        question_id = f"engine_mc_{index}"

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }
