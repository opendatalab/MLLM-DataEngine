import os
import sys
import pdb
import copy
import json
import tqdm
import logging
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist

from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import get_world_size, get_rank, is_dist_avail_and_initialized
from minigpt4.tasks.base_task import BaseTask
from minigpt4.common.logger import MetricLogger
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.aokvqa_tools import format_aokvqa_format, load_aokvqa, calculate_result
from minigpt4.tasks.utils import convert_dict_to_tensor
    
@registry.register_task("vqa_blip2")
class VQABlip2Task(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, vis_processor, anno_path, vis_root, save_result):
        mc_da = self.evaluate_on_AOKVQA(model, vis_processor, anno_path, vis_root, save_result)
        return mc_da
        
    def after_evaluation(self, val_result, split_name, epoch):
        val_result.update({"agg_metrics":val_result["DA"]})
        return val_result
        
    def evaluate_on_AOKVQA(self, model, vis_processor, anno_path, vis_root, save_result):
        def make_choice_text(choices, correct_choice_idx):
            letters = "ABCDEFG"
            choice_txt = ""
            for i in range(len(choices)):
                choice_txt += f"({letters[i]}) {choices[i]} "
            correct_letter = f"({letters[correct_choice_idx]})"
            return choice_txt.strip(), correct_letter
        
        print("Evaluating on AOKVQA...")
        aokvqa_val = json.load(open(anno_path, "r"))
        rank, word_size = get_rank(), get_world_size()
        step = len(aokvqa_val) // word_size + 1
        start, end = rank * step, (rank + 1) * step
        if end > len(aokvqa_val):
            end = len(aokvqa_val) - 1
        
        samples_all = aokvqa_val[start: end]
        samples_all = sorted(samples_all, key = lambda q:q["question_id"])
        results = []
        
        for index in tqdm.tqdm(range(len(samples_all))):
            image_id = samples_all[index]["image_id"]
            question = samples_all[index]["question"]
            choices = samples_all[index]["choices"]
            question_id = samples_all[index]["question_id"]
            correct_choice_idx = samples_all[index]["correct_choice_idx"]
            difficult_direct_answer = samples_all[index]["difficult_direct_answer"]
            direct_answers = samples_all[index]["direct_answers"]
            
            choice_txt, _ = make_choice_text(choices, correct_choice_idx)
            
            image = os.path.join(vis_root, f"{str(image_id).zfill(12)}.jpg")
            image = Image.open(image).convert('RGB')
            image = vis_processor(image).unsqueeze(0).cuda()
            
            _prompt = f"###Human: <img><ImageHere></img> {question}\n{choice_txt}\n###Assistant: The answer is"
            
            # encode image
            image_emb, image_atts = model.encode_img(image)
            
            # wrap image
            prompt_segs = _prompt.split('<ImageHere>')
            before = model.llama_tokenizer(
                prompt_segs[0], return_tensors="pt", add_special_tokens=False
            ).input_ids.cuda()
            after = model.llama_tokenizer(
                prompt_segs[1], return_tensors="pt", add_special_tokens=False
            ).input_ids.cuda()
            seg_tokens = [before, after]
            
            seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
            embs = torch.cat([seg_embs[0], image_emb, seg_embs[1]], dim=1)
            embed_atts = image_atts[:, :1].expand(-1, embs.shape[1])
            # embed attention mask & target
            embed_targets = torch.ones((embs.shape[0], embs.shape[1])).to(embs.device) * -100
            
            # embeds for answers
            loss_list = []
            for i, answer in enumerate(choices):
                letter = "ABCDEF"[i]
                _answer = f"({letter}) {answer}"
                to_regress_tokens = model.llama_tokenizer(
                    [_answer],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False
                ).to(embs.device)
                to_regress_embeds = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids.long())
                to_regress_target = copy.deepcopy(to_regress_tokens.input_ids)
                to_regress_atts = to_regress_tokens.attention_mask
                # no bos token in BLIP2
                embeds = torch.cat([embs, to_regress_embeds], dim=1)
                attention_mask = torch.cat([embed_atts, to_regress_atts], dim=1)
                target = torch.cat([embed_targets, to_regress_target], dim=1).long()
                
                with model.maybe_autocast():
                    outputs = model.llama_model(
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=target,
                    )
                loss_list.append(outputs['loss'])
            loss_list = torch.Tensor(loss_list)
            min_loss_idx = torch.min(loss_list, 0)[1].item()
            message = choices[min_loss_idx]
            
            results.append(
                {
                    "image_id": image_id,
                    "idx": question_id,
                    "question": question,
                    "choices": choices,
                    "correct_choice_idx": correct_choice_idx,
                    "direct_answers": direct_answers,
                    "difficult_direct_answer": difficult_direct_answer,
                    "model_answer": message,
                }
            )
        
        # convert dictionary -> tensor for gather all results in all ranks
        part_tensor = convert_dict_to_tensor(results)
        shape_tensor = torch.tensor(part_tensor.shape).cuda()
        shape_list = [shape_tensor.clone() for _ in range(get_world_size())]
        dist.all_gather(shape_list, shape_tensor)
    
        # gather tensor
        max_shape = max(shape_list)
        part_tensor_pad = torch.zeros(max_shape).cuda()
        part_tensor_pad[:part_tensor.shape[0]] = part_tensor
        tensor_list = [part_tensor_pad.clone() for _ in range(get_world_size())]
        dist.all_gather(tensor_list, part_tensor_pad)
    
        results_all_rank = []
        for tensor, shape in zip(tensor_list, shape_list):
            t = tensor.long()[:shape]
            _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
            _data = json.loads(_data)
            results_all_rank.extend(_data)
            
        # save model
        if save_result:
            filename = "evaluation/aokvqa_eval.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(results_all_rank, f)
                
        # gather and calculate results
        formatted_data = format_aokvqa_format(results_all_rank)
        dataset = json.load(open(anno_path))
        mc_da = calculate_result(predictions=formatted_data, dataset=dataset)
        logging.info(mc_da)
        
        return mc_da