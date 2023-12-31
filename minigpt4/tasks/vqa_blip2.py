import os
import sys
import pdb
import copy
import json
import tqdm
import torch
import logging
import numpy as np
from PIL import Image
import torch.distributed as dist

from minigpt4.common.registry import registry
from minigpt4.common.logger import MetricLogger
from minigpt4.common.dist_utils import get_world_size, get_rank, is_dist_avail_and_initialized
from minigpt4.tasks.aokvqa_tools import format_aokvqa_format, load_aokvqa, calculate_result
from minigpt4.tasks.utils import convert_dict_to_tensor
from minigpt4.tasks.base_task import BaseTask
from minigpt4.datasets.data_utils import prepare_sample
    
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
            question_type = samples_all[index]["question_type"]
            prompt = samples_all[index]["prompt"]
            
            choice_txt, _ = make_choice_text(choices, correct_choice_idx)
            
            image = os.path.join(vis_root, f"{str(image_id).zfill(12)}.jpg")
            image = Image.open(image).convert('RGB')
            image = vis_processor(image).unsqueeze(0).cuda()
            
            _prompt = f"###Human: <img><ImageHere></img> {question}\n{choice_txt}\n###Assistant: The answer is"
            
            image_emb, image_atts = model.encode_img(image)
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
            embed_targets = torch.ones((embs.shape[0], embs.shape[1])).to(embs.device) * -100
            
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
                    "image_path": os.path.join("val2017", f"{str(image_id).zfill(12)}.jpg"),
                    "idx": question_id,
                    "question": question,
                    "choices": choices,
                    "correct_choice_idx": correct_choice_idx,
                    "direct_answers": direct_answers,
                    "difficult_direct_answer": difficult_direct_answer,
                    "model_answer": message,
                    "question_type": question_type,
                    "prompt": prompt,
                }
            )
        
        # gather results of all ranks
        part_tensor = convert_dict_to_tensor(results)
        shape_tensor = torch.tensor(part_tensor.shape).cuda()
        shape_list = [shape_tensor.clone() for _ in range(get_world_size())]
        dist.all_gather(shape_list, shape_tensor)
    
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
            
        # save evaluation results
        if save_result:
            filename = "engine_pipeline/data/aokvqa_eval.json"
            print(f"save evaluation results to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(results_all_rank, f)
            
            results_all_rank_classified = {}
            for item in results_all_rank:
                if item["question_type"] not in results_all_rank_classified:
                    results_all_rank_classified[item["question_type"]] = []
                results_all_rank_classified[item["question_type"]].append(item)
                
            # save classified bad cases
            bad_case = "engine_pipeline/data/bad_case_aokvqa_classified.json"
            print(f"save classified bad case to {bad_case}")
            results_bad_case_classified = copy.deepcopy(results_all_rank_classified)
            for k in results_bad_case_classified.keys():
                results_bad_case_classified[k] = [
                    item for item in results_bad_case_classified[k]
                    if item["model_answer"] != item["choices"][item["correct_choice_idx"]]
                ]
            os.makedirs(os.path.dirname(bad_case), exist_ok=True)
            with open(bad_case, "w") as f:
                json.dump(results_bad_case_classified, f)
            
            # save weights
            weight_file = "engine_pipeline/data/weight.json"
            print(f"save weight of each question type to {weight_file}")
            type_weights = {}
            for k in results_all_rank_classified.keys():
                type_results = results_all_rank_classified[k]
                count_wrong = sum([1 for item in type_results if item["model_answer"] != item["choices"][item["correct_choice_idx"]]])
                type_weights[k] = round(count_wrong/len(type_results), 3)
            os.makedirs(os.path.dirname(weight_file), exist_ok=True)
            with open(weight_file, "w") as f:
                json.dump(type_weights, f)
                
        # gather and calculate results
        formatted_data = format_aokvqa_format(results_all_rank)
        dataset = json.load(open(anno_path))
        mc_da = calculate_result(predictions=formatted_data, dataset=dataset)
        logging.info(mc_da)
        
        return mc_da