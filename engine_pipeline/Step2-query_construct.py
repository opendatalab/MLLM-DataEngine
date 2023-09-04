# generate random prompt
import os
import json
import clip
import torch
import pickle
import random
from datetime import datetime
random.seed(datetime.now().timestamp())

from PIL import Image

import torch
import torch.nn.functional as F
import argparse


def process(COCO_root_path, bad_case_path, coco_embedding_path, qtype_distribution, MAXNUM, TOPK):
    PROMPT_TEMPLATE = """
    You are an AI visual assistant that can generate certain type of high-quality Q&A about the images. You will always perform as if you are directly seeing an image. 

    Goals: 
    I will give you some information about this image, along with a question type. Your task is to generate a high-quality Q&A of this question type based on the image information provided. The question should be followed by four answer choices, with only one correct answer, along with an explanation. 

    The image information includes 5 descriptions and the locations of specific objects within the image given in the form of bounding boxes. The bounding boxes coordinates are represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. 

    Rules (If not specified, the following rules apply to all texts you generate): 
    (1) Summarize the information provided by image descriptions and objects locations of this image to generate the high-quality Q&A, which should reflect the content of the image. 
    (2) Don't generate imaginative or irrelevant content. Ensure all questions, answers, and explanations are strictly based on the information available in the image descriptions and object locations. Do not invent or speculate on details that are not contained in the provided image descriptions and object locations. Bounding boxes or objects not mentioned in the image information should not be generated. 
    (3) You must always generate your responses as if you are directly viewing the image, not reading the image descriptions or object locations. Do not mention or refer to the image descriptions in your responses. Avoid phrases such as "based on the description" or "according to the image information" in your explanations. Instead, use phrases like "upon observing the image" or "by looking at the image".  
    (4) Do not ask questions that you cannot answer accurately with the given information. 
    (5) In the choices, there should be only one correct answer to the question. At the same time, other choices should also be relevant to the question. 
    (6) The question you ask should be also answerable without choices. 
    (7) Be sure to structure your questions and answers according to the question type. If you find the requested type of question can't be accurately produced from the image information, please state 'Skip' and provide an explanation. 
    (8) Don't give any clues or given conditions in question. Invoke visual information as much as possible. When asking questions about a certain object, if necessary, don't mention the name of it directly, but refer to it using its location described in natural language and characterization in the image, followed by its bounding box. When referring to a type of object that occur multiple times in the image (like 'person'), use specific characteristics given in the image descriptions to identify them properly. 
    (9) The bounding boxes should not be used as the main identifying feature of the objects in the scene, but rather as a supplement to descriptive identifiers.
    (10) Don't ask questions about object size or distance unless there's ample evidence in the image descriptions about object scale or spatial relations. Bounding box details should not be used to infer the sizes or shapes of the objects, becase it is in 2D but the the objects are in complex 3D scenes. Instead of asking "Which is larger?", you can ask "Which object occupies a larger area of the image?" 
    (11) Avoid questions that might be confusing due to similar objects. If bounding boxes are too small or numerous, the AI should avoid forming questions about those objects to avoid confusion. 
    (12) Do not mention or refer to the image descriptions in your responses.

    Here is an example: 
    Image information: 
    (1) Image description: 
    A photo of two youth soccer teams competing. 
    Little kids play a game of soccer in a field. 
    A group of young men kicking around a ball. 
    The small children play soccer on a sunny day. 
    Children run after a soccer ball in the grass. 

    (2) Objects locations, in the form of bounding box (object: [x1, y1, x2, y2]): 
    sports ball: [0.324, 0.769, 0.44, 0.933] 
    person: [0.003, 0.011, 0.202, 0.793] 
    person: [0.125, 0.053, 0.414, 0.868] 
    person: [0.41, 0.001, 0.658, 0.886] 
    person: [0.36, 0.002, 0.476, 0.329] 
    person: [0.187, 0.0, 0.416, 0.258] 
    person: [0.965, 0.003, 1.0, 0.219] 

    Question: Who is more likely to kick the sports ball [0.324, 0.769, 0.44, 0.933] next? 
    Choices: (A) The person located to the far left side of the image [0.003, 0.011, 0.202, 0.793] (B) The person located closer to the center of the image [0.125, 0.053, 0.414, 0.868] (C) The person located to the far right side of the image [0.965, 0.003, 1.0, 0.219] (D) The person located closer to the bottom of the image [0.41, 0.001, 0.658, 0.886] 
    Answer: The answer is (B): The person located closer to the center of the image [0.125, 0.053, 0.414, 0.868]. 
    Explanations: Based on the distance between the soccer players and the ball in the image, the player closer to the center of the image [0.125, 0.053, 0.414, 0.868] appears to be in the best position to make the next move on the sports ball [0.324, 0.769, 0.44, 0.933]. The player to the far right [0.965, 0.003, 1.0, 0.219] seems too far away, and the players at the bottom [0.41, 0.001, 0.658, 0.886] and far left [0.003, 0.011, 0.202, 0.793] of the image look to be at a disadvantageous position to kick the ball next. 

    Here is the Image information that I want you to generate high-quality {} Q&A from: 
    (1) Image description: 
    {} 

    (2) Objects locations, in the form of bounding box (object: [x1, y1, x2, y2]): 
    {} 

    The question type I require is {} question. {} Following are two examples of the {} question: 
    Example 1: 
    {} 
    Example 2: 
    {} 

    Now you can start to generate one high-quality {} Q&A about the image according to the image information I gave you.

    """


    qtype_explanation = {
        "Identity reasoning": "Predict the identity of a person. Example: by observing a person’s clothing and appearance, one may infer his / her occupation and social status.",
        "Physical property reasoning": "Predict the physical property of an object. Examples: the physical property of concentrated sulfuric acid is that it is volatile, the physical property of water is its fluidity, etc.",
        "Attribute recognition": "Recognition of texture, shape, appearance characteristics, emotions, category, celebrities, famous places and objects, optical characters.",
        "Function reasoning": "Predict the function of an object. Examples: the function of a broom is to sweep the floor, the function of a spatula is to cook, the function of a pen is to write, etc.",
        "Object localization": "For a single object, determine its position in the image (such as top, bottom, etc.), its absolute coordinates in the image, count the number of objects, and the orientation of the object.",
        "Structuralized imagetext understanding": "Structured understanding of images and text, including parsing the content of charts (such as the trends of multiple bars in a bar chart), understanding the code in an image, etc.",
        "Attribute comparison": "Compare attributes of different objects in image, such as shape, color, etc.",
        "Nature relation": "Other abstract relationships that exist in nature. Examples: predation, symbiosis, coexistence, etc.",
        "Future prediction": "Predict what will happen in the future. Examples: if it is thundering in the sky now, it can be predicted that it will rain soon (physical phenomenon); if someone raises their fist, it means they are going to hit someone (event occurrence); if someone’s face becomes serious, it means they are going to get angry (emotional change).",
        "Image scene": "Determine which environment is shown in the image, such as indoors, outdoors, forest, city, mountains, waterfront, sunny day, rainy day, etc.",
        "Spatial relationship": "Determine the relative position between objects in image.",
        "Image quality": "Determine the objective quality of the image, such as whether it is blurry, bright or dark, contrast, etc.",
        "Physical relation": "All relationships that exist in the physical world, 3D spatial relationships and the connections between objects are.",
        "Action recognition": "Recognizing human actions, including pose motion, human-object interaction, and human-human interaction.",
        "Ocr": "Recognition of text, formula, and sheet in the image.",
        "Social relation": "Relations in human society or relations defined from the human perspective. Examples: Inter-person relations, such as father and son, husband and wife, friend, hostile, etc.",
        "Celebrity recognition": "Recognition of celebrities, landmarks, and common items.",
        "Image style": "Determine which type of image it belongs to, such as photos, paintings, CT scans, etc.",
        "Image emotion": "Determine which subjective emotion is conveyed by the overall image, such as cold, cheerful, sad, or oppressive.",
        "Knowledge-based reasoning": "Require pre-existing knowledge outside the content of the image. Example: the year of this object invented, the top ranked player in this sport, etc.", 
        "Image topic": "Determine what the subject of the image is, such as scenery, portrait, close-up of an object, text, etc."
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load offline coco embeddings
    print("loading coco embeddings")
    with open(coco_embedding_path, "rb") as f:
        coco_embed = pickle.load(f)
    image_keys = list(coco_embed.keys())
    image_embeds = list(coco_embed.values())
    image_embeds = torch.cat(image_embeds, dim=0).to(device)
    print(image_embeds.shape)
        
    # load coco annotations
    train2017 = json.load(open(os.path.join(COCO_root_path, "annotations", "instances_train2017.json"), "r"))
    id2cat = {cat["id"]:cat["name"] for cat in train2017["categories"]}
    anno_all, hw_all, file_all = {}, {}, {}
    # annotations
    for anno in train2017["annotations"]:
        image_id = anno["image_id"]
        bbox = anno["bbox"]
        cat  = id2cat[anno["category_id"]]
        if image_id not in anno_all:
            anno_all[image_id] = []
        if bbox+[cat] not in anno_all[image_id]:
            anno_all[image_id].append(bbox+[cat])
    # width & height
    for image in train2017["images"]:
        hw_all[image["id"]] = [image["width"], image["height"]]
    # file
    for image in train2017["images"]:
        file_all[image["id"]] = os.path.join(COCO_root_path, "train2017", image["file_name"])
    # image caption
    coco_cap = json.load(open(os.path.join(COCO_root_path, "annotations", "captions_train2017.json"), "r"))
    cap_all = {}
    for anno in coco_cap["annotations"]:
        if anno["image_id"] not in cap_all:
            cap_all[anno["image_id"]] = []
        if anno["caption"] not in cap_all[anno["image_id"]]:
            cap_all[anno["image_id"]].append(anno["caption"])
    
    # load coco annotations
    val2017 = json.load(open(os.path.join(COCO_root_path, "annotations", "instances_val2017.json"), "r"))
    val_file_all = {}
    for image in val2017["images"]:
        val_file_all[image["id"]] = os.path.join(COCO_root_path, "val2017", image["file_name"])

    def get_bbox_desc(image_id):
        # bounding box description
        w, h = hw_all[int(image_id)]
        bbox_desc = ""
        try:
            if len(anno_all[int(image_id)]) > 10:
                return 
            for i, anno in enumerate(anno_all[int(image_id)]):
                cat, bbox = anno[-1], anno[:4]
                bbox = [bbox[0]/w, bbox[1]/h, (bbox[0]+bbox[2])/w, (bbox[1]+bbox[3])/h]
                bbox = [round(b,3) for b in bbox]
                #bbox_desc += f"{cat} <obj_{i+1}>: {bbox}\n"
                bbox_desc += f"{cat} : {bbox}\n"
        except BaseException:
            pass
        return bbox_desc

    def get_cap(image_id):
        cap_text = ""
        try:
            for cap in cap_all[int(image_id)]:
                cap_text += cap + "\n"
        except BaseException:
            pass
        return cap_text
        
    # load clip model
    print("load clip...")
    model, preprocess = clip.load("ViT-B/32", device=device)
        
    # bad case pool
    bad_case_pool = json.load(open(bad_case_path, "r"))

    #del bad_case_pool["structuralized_imagetext_understanding"]
    #del bad_case_pool["ocr"]
    #del bad_case_pool["celebrity_recognition"]
    qtype_len = [len(l) for l in list(bad_case_pool.values())]

    #qtype_distribution = [l/sum(qtype_len) for l in qtype_len]
    if not qtype_distribution:
        qtype_distribution = {
        'action_recognition': 1,
        'attribute_comparison': 1,
        'attribute_recognition': 1,
        'celebrity_recognition': 1,
        'function_reasoning': 1,
        'future_prediction': 1,
        'identity_reasoning': 1,
        'image_emotion': 1,
        'image_quality': 1,
        'image_scene': 1,
        'image_style': 1,
        'image_topic': 1,
        'nature_relation': 1,
        'object_localization': 1,
        'ocr': 1,
        'physical_property_reasoning': 1,
        'physical_relation': 1,
        'social_relation': 1,
        'spatial_relationship': 1,
        'structuralized_imagetext_understanding': 1,
        'knowledge-based_reasoning': 1,
        }

    filter_image_ids = []   # put image ID needs to be filtered

    filter_image_ids = list(set(filter_image_ids))

    weights = [float(qtype_distribution[qtype]) for qtype in bad_case_pool.keys()]
    print('question type list: ', list(bad_case_pool.keys()))
    print('weight list: ', weights)
    

    cnt = 0
    select_data = []
    select_data_2 = []
    #for index in range(MAXNUM):
    while cnt < MAXNUM:
        if cnt % 1000 == 0:
            print(f"{cnt}/{MAXNUM}")
        # random a question type
        qtype_p = random.choices(
            population = list(bad_case_pool.keys()),
            weights = weights,
            k=1
        )[0]
        
        # random 2 in-context learning samples
        # random 1 sample as query
        
        if len(bad_case_pool[qtype_p]) < 2:
            continue
        
        try:
            in_context_samples = random.sample(bad_case_pool[qtype_p], 2)
        except BaseException:
            in_context_sample = bad_case_pool[qtype_p][0]
            in_context_samples = [in_context_sample, in_context_sample]
        
        anchor_sample = in_context_samples[0]
        #print(anchor_sample)
        
        # anchor image
        if cnt%2 == 0:
            if anchor_sample.get("image_path", None) is not None:
                image_path = os.path.join(COCO_root_path, anchor_sample["image_path"])
            else:
                image_path = val_file_all[anchor_sample["image_id"]]
            anchor_image = Image.open(image_path)
            #anchor_image.save("anchor.jpg")
            
            with torch.inference_mode():
                anchor_image = preprocess(anchor_image).unsqueeze(0).to(device)
                anchor_features = model.encode_image(anchor_image)
        
                #print(anchor_features.shape)
                #print(image_embeds.shape)
        
                # similarity
                anchor_features = anchor_features.repeat(image_embeds.shape[0], 1)
                similarity = F.cosine_similarity(image_embeds, anchor_features)
                #print(similarity.shape)
        
                # topk
                topk_query = torch.topk(similarity, TOPK)
                topk_query_idx = topk_query[1]
                #print(topk_query_idx)
        
            rand_idx = torch.randperm(topk_query_idx.shape[0])
            topk_query_idx = topk_query_idx[rand_idx]
            query_image_id = image_keys[topk_query_idx[0]]
        else:
            query_image_id = random.choice(image_keys)
        
        if query_image_id not in filter_image_ids:
            filter_image_ids.append(query_image_id)
        else:
            continue
            
        bbox_desc = get_bbox_desc(query_image_id)
        if bbox_desc is None:
            continue
            
        cnt += 1
            
        cap = get_cap(query_image_id)
            
        explain = qtype_explanation[qtype_p.replace("_", " ").capitalize()]
        
        def get_prompt(sample):
            txt = "Question: {}\nChoices: (A) {} (B) {} (C) {} (D) {}\nAnswer: The answer is ({}): {}.\nQuestion type: {}".format(sample['question'], sample['choices'][0], sample['choices'][1], sample['choices'][2], sample['choices'][3], ["A","B","C","D"][sample['correct_choice_idx']], sample['choices'][sample['correct_choice_idx']], sample['question_type'].replace('_',' '))
            return txt
        prompt = PROMPT_TEMPLATE.format(
            qtype_p.replace("_", " "),
            cap,
            bbox_desc,
            qtype_p.replace("_", " "), 
            explain,
            qtype_p.replace("_", " "),
            in_context_samples[0].get('prompt', get_prompt(in_context_samples[0])),
            in_context_samples[1].get('prompt', get_prompt(in_context_samples[1])),
            qtype_p.replace("_", " "), 
        )
        
        select_data.append({
            "image_id": query_image_id,
            "prompt": prompt,
            "question_type": qtype_p,
        })

    return select_data
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT-based QA classification.')
    parser.add_argument('-i', '--input')
    parser.add_argument('-c', '--COCO_dataset')
    parser.add_argument('-e', '--COCO_embeding')
    parser.add_argument('-w', '--weight', default="")
    parser.add_argument('-m', '--maxnum', default=4500)
    parser.add_argument('-t', '--topk', default=1000)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    MAXNUM = int(args.maxnum)
    TOPK = int(args.topk)

    if args.weight:
        with open(args.weight, 'r') as f:
            qtype_distribution = json.load(f)

    select_data = process(args.COCO_dataset, args.input, args.COCO_embeding, qtype_distribution, MAXNUM, TOPK)

    with open(args.output, "w") as f:
        json.dump(select_data, f)
