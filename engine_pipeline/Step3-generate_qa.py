from typing import List
import json
import os
import random
import re
import string
import jsonlines
import tqdm
import utils
import os
from collections import defaultdict
import time
import sys
import logging
import argparse

def generate_instruction_following_data(
    sample=[],
    model_name="gpt-4",
    temperature=1.0,
    top_p=1.0,
):
    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=800,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        stop=["###"],
    )

    results = utils.openai_completion(
        prompts=[sample['prompt']],
        model_name=model_name,
        batch_size=1,
        use_chat=True,
        decoding_args=decoding_args,
        logit_bias={
            "50256": -100
        },  # prevent the <|endoftext|> token from being generated
    )
    
    miss_samples = []

    response = results[0]
    if response is None:
        miss_samples = sample
    
    # raw_instructions = post_process_gpt35(response)
    raw_instructions = response["message"]["content"].strip()

    instructions = {
        "image_id": sample['image_id'],
        "raw_instructions": raw_instructions,
        "question_type": sample['question_type']
    }
    
    return instructions, miss_samples


if __name__ == "__main__":
    """
    you may need to change the param below:
        path: data path
        model_name : "gpt-3.5-turbo" or "text-davinci-003" or "gpt-4"
    """

    # Get the list of all files and directories
    parser = argparse.ArgumentParser(description='GPT-based QA generation.')
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        dir_list = json.load(f)
    model_name = "gpt-4"
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    output_file = args.output
    output_miss = output_dir.split('.')[0] + "_miss_id.jsonl"
    info_path = args.input

    # Ensure continued operation from the breakpoint after reconnection after disconnection
    if os.path.exists(output_file):
        with open(output_file, 'r') as ff:
            finished = ff.readlines()
            start_idx = len(finished)
            check_point = json.loads(finished[-1])
            match_point = dir_list[start_idx-1]
            if check_point['image_id'] != match_point['image_id']:
                print(f"The check point {check_point} not match the match point {match_point}, the start id should not be {start_idx}. Check.")
                sys.exit()
    else:
        start_idx = 0  # 开始的index

    # now let's generate new instructions
    batch_size = 1
    end_idx = len(dir_list)
    # end_idx = 5
    progress_bar = tqdm.tqdm(
        range(start_idx, end_idx, batch_size), desc="images"
    )
    print(f"The process for {info_path} will be start from {start_idx} to {end_idx} with batch size {batch_size}, the ouput path is {output_file}.")

    
    for i in progress_bar:
        sample = dir_list[i]
        res, miss_id = generate_instruction_following_data(
            sample=sample,
            model_name=model_name,
            temperature=0.7,
            top_p=0.9,
        )

        with open(output_file, "a") as outfile:
            json.dump(res, outfile)
            outfile.write("\n")
        # Storing problematic image IDs for later rerun if needed
        if miss_id:
            with open(output_miss, "a") as outfile:
                json.dump(miss_id, outfile)
                outfile.write("\n")
        time.sleep(5)
