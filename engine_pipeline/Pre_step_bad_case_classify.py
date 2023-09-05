import os
import re
import sys
import copy
import math
import openai
import logging
import dataclasses
from openai import openai_object
from typing import Optional, Sequence, Union
import time
import utils
import argparse


def GPT_classify(bad_cases):

    prompt_temp = """You are an AI assistant that can do text categorization.

    Goals:
    I will give you some questions, you should classify them into one of the following question types:
    (1) Identity reasoning: Predict the identity of a person. Example: by observing a person’s clothing and appearance, one may infer his / her occupation and social status. 
    (2) Physical property reasoning: Predict the physical property of an object. Examples: the physical property of concentrated sulfuric acid is that it is volatile, the physical property of water is its fluidity, etc. 
    (3) Attribute recognition: Recognition of texture, shape, appearance characteristics, emotions, category, celebrities, famous places and objects, optical characters. 
    (4) Function reasoning: Predict the function of an object. Examples: the function of a broom is to sweep the floor, the function of a spatula is to cook, the function of a pen is to write, etc. 
    (5) Object localization: For a single object, determine its position in the image (such as top, bottom, etc.), its absolute coordinates in the image, count the number of objects, and the orientation of the object. 
    (6) Attribute comparison: Compare attributes of different objects in image, such as shape, color, etc. 
    (7) Nature relation: Other abstract relationships that exist in nature. Examples: predation, symbiosis, coexistence, etc. 
    (8) Future prediction: Predict what will happen in the future. Examples: if it is thundering in the sky now, it can be predicted that it will rain soon (physical phenomenon); if someone raises their fist, it means they are going to hit someone (event occurrence); if someone’s face becomes serious, it means they are going to get angry (emotional change). 
    (9) Image scene: Determine which environment is shown in the image, such as indoors, outdoors, forest, city, mountains, waterfront, sunny day, rainy day, etc. 
    (10) Spatial relationship: Determine the relative position between objects in image. 
    (11) Image quality: Determine the objective quality of the image, such as whether it is blurry, bright or dark, contrast, etc. 
    (12) Physical relation: All relationships that exist in the physical world, 3D spatial relationships and the connections between objects are. 
    (13) Action recognition: Recognizing human actions, including pose motion, human-object interaction, and human-human interaction. 
    (14) Social relation: Relations in human society or relations defined from the human perspective. Examples: Inter-person relations, such as father and son, husband and wife, friend, hostile, etc. 
    (15) Image style: Determine which type of image it belongs to, such as photos, paintings, CT scans, etc. 
    (16) Image emotion: Determine which subjective emotion is conveyed by the overall image, such as cold, cheerful, sad, or oppressive. 
    (17) Image topic: Determine what the subject of the image is, such as scenery, portrait, close-up of an object, text, etc. 
    (18) Knowledge-based reasoning: Require pre-existing knowledge outside the content of the image. Example: the year of this object invented, the top ranked player in this sport, etc. 

    The questions to be classified:
    {}
    Here is an example of your response:
    1. (6) Physical relation
    2. (15) OCR
    3. ...
    """

    decoding_args = utils.OpenAIDecodingArguments(
        temperature=0.7,
        n=1,
        max_tokens=800,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=0.9,
        stop=["###"],
    )

    # print(prompt_temp)

    result_list = []
    for i in range(0, len(bad_cases), 50):
        print(f'---------Process cases from {i} to ', i+len(bad_cases[i:i+50]))
        que_txt = ""
        for idx, case in enumerate(bad_cases[i:i+50]):
            que_txt += str(idx+1) + '. ' + case['question'] + '\n'
            
        prompt = prompt_temp.format(que_txt)
        results = utils.openai_completion(
            prompts=[prompt],   # [1+2+3,..]
            model_name="gpt-4",
            batch_size=1,
            use_chat=True,
            decoding_args=decoding_args,
            logit_bias={
                "50256": -100
            },  # prevent the <|endoftext|> token from being generated
        )
        
        result_list.append(results[0]["message"]["content"].strip())
    return result_list


def post_process(bad_cases, result_list, image_root_path):
    # post-process
    qtype = """(1) identity reasoning
    (2) physical property reasoning
    (3) attribute recognition
    (4) function reasoning
    (5) object localization
    (6) attribute comparison
    (7) nature relation
    (8) future prediction
    (9) image scene
    (10) spatial relationship
    (11) image quality
    (12) physical relation
    (13) action recognition
    (14) social relation
    (15) image style
    (16) image emotion
    (17) Image topic
    (18) Knowledge-based reasoning"""

    qtype_list = []
    for line in qtype.split('\n'):
        txt = line.split(')')[1].strip().replace(' ','_').lower()
        qtype_list.append(txt)
    print(qtype_list)

    classified_cases = []
    batch_id = 0
    for i in range(0, len(bad_cases), 50):
        case_batch = copy.deepcopy(bad_cases[i:i+50])
        result = result_list[batch_id]
        wrong_flag = False
        print(f'Process cases from {i} to ', i+len(case_batch))
        if len(result.split('\n')) == len(case_batch):
            for idx, line in enumerate(result.split('\n')):
                if int(line.split('.')[0].strip())==idx+1:
                    num = int(line.split('.')[1].split('(')[1].split(')')[0].strip())
                    txt = line.split('.')[1].split('(')[1].split(')')[1].strip().replace(' ','_').lower()
                    # if qtype_list[num-1] == txt:
                    if txt in qtype_list:
                        case_batch[idx]['question_type'] = txt
                    else:
                        print(f'wrong qtype in line {idx}: {txt}')
                        wrong_flag = True
                else:
                    print('wrong idx in line {idx}: {num}')
                    wrong_flag = True
        else:
            print('wrong input len', len(result.split('\n')))
            wrong_flag = True
        
        if wrong_flag:
            print(result)
            sys.exit()
        else:
            classified_cases.extend(case_batch)
            batch_id += 1

    qtype_dict = {}
    for qtype in qtype_list:
        qtype_dict[qtype] = []

    for sample in classified_cases:
        txt = "Question: {}\nChoices: (A) {} (B) {} (C) {} (D) {}\nAnswer: The answer is ({}): {}.\nQuestion type: {}".format(sample['question'], sample['choices'][0], sample['choices'][1], sample['choices'][2], sample['choices'][3], ["A","B","C","D"][sample['correct_choice_idx']], sample['choices'][sample['correct_choice_idx']], sample['question_type'].replace('_',' '))
        sample['prompt'] = txt
        sample['image_path'] = os.path.join(image_root_path, str(sample['image_id']).zfill(12) + '.jpg')
        qtype_dict[sample['question_type']].append(sample)
    
    return qtype_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-based QA classification.')
    parser.add_argument('-i', '--input')
    parser.add_argument('-m', '--image')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    image_root_path = args.image

    # GPT response
    import json
    with open(args.input, 'r') as f:
        bad_cases = json.load(f)

    result_list = GPT_classify(bad_cases)

    qtype_dict = post_process(bad_cases, result_list, image_root_path)

    with open(args.output, 'w') as f:
        json.dump(qtype_dict, f)