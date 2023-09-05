import torch
import torch.nn as nn
import argparse
from transformers import LlamaTokenizer
from minigpt4.models.modeling_llama import LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="apply lora delta")
    parser.add_argument("--base-model", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt", required=True, help="ckpt that stores lora weight.")
    parser.add_argument("--lora-scale", required=False, type=float, default=2, help="lora scaling.")
    parser.add_argument("--target", required=True, help="target path that saves model.")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    # load base model
    # load torch.float32 to avoid loss precision
    print("load base model...")
    llama_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    )
    llama_tokenizer = LlamaTokenizer.from_pretrained(args.base_model, use_fast=False)
    
    # load lora delta
    print("load lora delta...")
    lora_delta = torch.load(args.ckpt, map_location="cpu")
    
    for name, param in llama_model.named_parameters():
        param.requires_grad = False
    
    # merge weights
    print("apply lora delta...")
    for idx in range(len(llama_model.model.layers)):
        llama_model.model.layers[idx].self_attn.q_proj.weight = nn.Parameter(
            llama_model.model.layers[idx].self_attn.q_proj.weight + (torch.matmul(
                lora_delta["model"][f"llama_model.model.layers.{idx}.self_attn.q_proj.lora_B.weight"],
                lora_delta["model"][f"llama_model.model.layers.{idx}.self_attn.q_proj.lora_A.weight"]
            ) * args.lora_scale)
        )
        llama_model.model.layers[idx].self_attn.k_proj.weight = nn.Parameter(
            llama_model.model.layers[idx].self_attn.k_proj.weight + (torch.matmul(
                lora_delta["model"][f"llama_model.model.layers.{idx}.self_attn.k_proj.lora_B.weight"],
                lora_delta["model"][f"llama_model.model.layers.{idx}.self_attn.k_proj.lora_A.weight"]
            ) * args.lora_scale)
        )
        
    # save merged model
    print("save merged model...")
    llama_model.save_pretrained(args.target)
    llama_tokenizer.save_pretrained(args.target)