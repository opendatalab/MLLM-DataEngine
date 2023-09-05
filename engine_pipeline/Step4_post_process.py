import re
import os
import json
import random
from PIL import Image
from copy import deepcopy
import argparse

def located_at_bbox(lst):
    for s in lst:
        pattern = r"located at\s*\["
        if re.search(pattern, s):
            return True
    return False


def remove_blank(x):
    split = x.split(" ")
    split = [s for s in split if len(s)>0]
    return " ".join(split).strip()


# remove boungding boxes
def remove_bbox_v2(x):
    #print(x)
    x = re.sub("\[.*?\]","",x)
    x = re.sub('\s*and\s*([,.?])', r'\1', x)
    x = re.sub('(\s*,\s*)+', ', ', x)
    x = re.sub('\s*,\s*([.?])', r'\1', x)
    x = re.sub('\s*([,.?])', r'\1', x)
    return remove_blank(x)


def have_bbox(s):
    if "coordinates" in s or "coordinate" in s or "bounding box" in s:
        return True
    return False


def skip(s):
    if "skip" in s or "Skip" in s:
        return True
    return False


def process_data(data_all):
    print(f'init_len: {len(data_all)}',end='\t')

    # move coordinates or bounding box
    data_all = [_data for _data in data_all if not have_bbox(_data["raw_instructions"])]
    
    # move coordinates or bounding box
    data_all = [_data for _data in data_all if not skip(_data["raw_instructions"])]
    
    process_data_all = []
    for idx, _data in enumerate(data_all):
            
        # question
        index_q = _data["raw_instructions"].find("Question:")
        # multiple choices
        index_m = _data["raw_instructions"].find("Choices:")
        # answer
        index_a = _data["raw_instructions"].find("Answer:")
        # explaination
        index_e = _data["raw_instructions"].find("Explanation:")
        if index_e == -1:
            index_e = _data["raw_instructions"].find("Reason:")
        if index_e == -1:
            index_e = _data["raw_instructions"].find("Reasoning:")
        if index_e == -1:
            index_e = _data["raw_instructions"].find("Lecture:")
        if index_e == -1:
            index_e = _data["raw_instructions"].find("Explanatory note:")
        if index_e == -1:
            index_e = _data["raw_instructions"].find("Explaination:")
            
        question = _data["raw_instructions"][index_q+9:index_m].strip()
        
        multi_choice = _data["raw_instructions"][index_m+8:index_a].strip()
        choices = []
        for i, c1 in enumerate(multi_choice):
            if c1 == ")":
                for j, c2 in enumerate(multi_choice[i:]):
                    if c2 == "(":
                        break
                if i+j == len(multi_choice)-1:
                    choices.append(multi_choice[i+1:i+j+1].strip())
                else:
                    choices.append(multi_choice[i+1:i+j].strip())
        choices = [c for c in choices if len(c)>0]
                    
        for i, c in enumerate(choices):
            if c[-1] == ".":
                choices[i] = c[:-1]
        
        try:
            assert len(choices) > 0
        except AssertionError:
            continue
        
        assert all([len(c) != 0 for c in choices])
        
        answer = _data["raw_instructions"][index_a:index_e].strip()
        try:
            res = answer.split("(")[1].split(")")[0].upper()
        except IndexError:
            continue
        
        assert res in ["A", "B", "C", "D", "E"]
        
        correct_answer_idx = ["A", "B", "C", "D", "E"].index(res)
        assert correct_answer_idx != -1
        
        explain = _data["raw_instructions"][index_e:].strip()
            
        if index_q == -1 or index_m == -1 or index_a == -1 or index_e == -1:
            explain = _data["raw_instructions"].split("\n")[-1]
            
        _data.update({
            "question": question,
            "choices": choices,
            "correct_choice_idx": correct_answer_idx,
            "rationales": [explain],
        })
        process_data_all.append(_data)

    process_data_all_fin = []
    for data in process_data_all:
        if (located_at_bbox([data["question"]]) or located_at_bbox(data["choices"]) or located_at_bbox(data["rationales"])):
            continue
        _data=deepcopy(data)
        _data["question"] = remove_bbox_v2(_data["question"])
        _data["choices"] = [remove_bbox_v2(c) for c in _data["choices"]]
        _data["rationales"] = [remove_bbox_v2(r) for r in _data["rationales"]]
        process_data_all_fin.append(_data)

    process_data_all_no_desc=[]
    for d in process_data_all_fin:
        if "description" not in d['rationales'][0].lower():
            process_data_all_no_desc.append(d)
            
    print(f'processed_len: {len(process_data_all_no_desc)}')
    
    return process_data_all_no_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process.')
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.input, "r") as f:
        your_gptqa = []
        for line in f.readlines():
            your_gptqa.append(json.loads(line))
    
    your_gptqa_processed = process_data(your_gptqa)

    with open(args.output,'w') as f:
        json.dump(your_gptqa_processed,f,indent=2)