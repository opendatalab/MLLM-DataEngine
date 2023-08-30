import logging
import random
import copy

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import CLIPVisionModel

@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        llama_lora=None,
        freeze_llama_proj=False,
        fully_open=False,
        task_prompt=False,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        self.task_prompt = task_prompt
        
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_lora = False
        if llama_lora is not None:
            llama_freeze = llama_lora.pop('freeze')
            self.llama_freeze = llama_freeze
            setattr(LlamaForCausalLM, 'lora_cfg', llama_lora)
            self.llama_lora = True
        
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        elif self.llama_lora:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        elif fully_open:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float32,
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if not fully_open:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                if self.llama_lora and name.find("lora") != -1 and not self.llama_freeze:
                    param.requires_grad = True
            for name, module in self.llama_model.named_modules():
                if self.llama_lora and name.find("lora") != -1 and not self.llama_freeze:
                    module.float()
        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = True
                    
        print('Loading LLAMA Done')
        
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if freeze_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                print ("freeze {} for lora only finetuning".format(name))
                param.requires_grad = False
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_embeds).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def make_vqa_targets(self, to_regress_tokens, mask_text):
        targets = copy.deepcopy(to_regress_tokens.input_ids)
        mask_token = self.llama_tokenizer(mask_text, return_tensors="pt", padding="longest", truncation=True,
                                          max_length=self.max_txt_len, add_special_tokens=False).to(targets.device)
        mask_token_lens = [
            tokenized.ne(self.llama_tokenizer.pad_token_id).sum().item()
            for tokenized in mask_token.input_ids
        ]
        for mask_token_len, target in zip(mask_token_lens, targets):  #mask question + ### + Assistant:
            target[:mask_token_len] = -100

        targets = targets.masked_fill(targets == self.llama_tokenizer.pad_token_id, -100)
        return targets
        
    def make_conv_targets(self, to_regress_tokens):
        target_batch = []
        for bs in range(to_regress_tokens.input_ids.shape[0]):
            cur_idx = 0
            tokens = self.llama_tokenizer.convert_ids_to_tokens(to_regress_tokens.input_ids[bs])
            targets = torch.ones(to_regress_tokens.input_ids.shape[1]).long() * -100
            while cur_idx < len(tokens) and to_regress_tokens.input_ids[bs][cur_idx] != self.llama_tokenizer.pad_token_id:
                if tokens[cur_idx:cur_idx+5] == ['##', '#', 'Ass', 'istant', ':']:
                    no_mask_idx = cur_idx + 5
                    while no_mask_idx < len(tokens) and to_regress_tokens.input_ids[bs][no_mask_idx] != self.llama_tokenizer.pad_token_id:
                        if tokens[no_mask_idx:no_mask_idx+4] == ['##', '#', 'Human', ':']:
                            break
                        targets[no_mask_idx] = to_regress_tokens.input_ids[bs][no_mask_idx]
                        no_mask_idx += 1
                    cur_idx = no_mask_idx + 1
                else:
                    cur_idx += 1
                    continue
            target_batch.append(targets.unsqueeze(0))
        target_batch = torch.cat(target_batch, dim=0)
        target_batch = target_batch.masked_fill(target_batch == self.llama_tokenizer.pad_token_id, -100).to(to_regress_tokens.input_ids.device)
        return target_batch
        
    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if "data_type" in samples:
            if "vqa" in samples["data_type"][0]:  # VQA dataset
                if self.task_prompt:
                    if samples["data_type"][0] == "vqa":
                        vqa_prompt = '[Visual Question Answering] ###Human: <Img><ImageHere></Img> '
                    elif samples["data_type"][0] == "multi_choice_vqa":
                        vqa_prompt = '[Multi-choice Visual Question Answering] ###Human: <Img><ImageHere></Img> '
                else:
                    vqa_prompt = '###Human: <Img><ImageHere></Img> '
                #print(vqa_prompt)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            if samples["data_type"][0] == "conversation":
                conv_prompt = '###Human: <Img><ImageHere></Img> '
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, conv_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if self.task_prompt:
                prompt = "[Detailed Image Description] " + prompt
            #print(prompt)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        if "data_type" in samples:
            if "vqa" in samples["data_type"][0]:  # VQA dataset
                targets = self.make_vqa_targets(to_regress_tokens, samples['mask_text'])
            if samples["data_type"][0] == "conversation":
                targets = self.make_conv_targets(to_regress_tokens)
        else:
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")
        llama_lora = cfg.get('llama_lora')
        freeze_llama_proj = cfg.get('freeze_llama_proj')
        fully_open = cfg.get('fully_open', False)
        task_prompt = cfg.get("task_prompt", False)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            llama_lora=llama_lora,
            freeze_llama_proj=freeze_llama_proj,
            fully_open=fully_open,
            task_prompt=task_prompt,
        )
        
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                msg = model.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = model.load_state_dict(ckpt, strict=False)
            
        return model
