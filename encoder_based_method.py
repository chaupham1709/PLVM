import argparse

from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoImageProcessor, Dinov2Model
import copy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.conversation import conv_templates, SeparatorStyle
from packaging import version
import tokenizers
from accelerate.logging import get_logger
import torch.nn as nn
import os
import numpy as np
from torch.nn import functional as F
import math
import types
from typing import Union, List
from transformers import TextStreamer
from einops import rearrange
from einops.layers.torch import Rearrange
from accelerate.hooks import add_hook_to_module
from transformers.integrations import is_deepspeed_zero3_enabled
from llava.utils import disable_torch_init

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
PERSONALIZATION_TOKEN = "<sks>"
logger = get_logger(__name__)


def tokenizer_image_token(prompt, tokenizer, personalization_token_id, image_token_index=IMAGE_TOKEN_INDEX, personalization_token=PERSONALIZATION_TOKEN, num_soft_tokens=16, return_tensors="pt"):
    soft_prompt_ids = [personalization_token_id + i for i in range(1, num_soft_tokens+1)]
    if '<image>' in prompt:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
        sep_prompt_list = prompt.split('<image>')
        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        # Normally, a prompt consists 3 parts:
        # 1 System prompt.
        # <sks> is <soft prompt>.
        # USER: query from user and the corresponding answers
        sys_prompt = sep_prompt_list[0].split(personalization_token)[0].strip()
        input_ids.extend(tokenizer(sys_prompt).input_ids[offset:])
        # Insert <sks> is <soft prompt>.
        input_ids.append(personalization_token_id)
        input_ids.extend(tokenizer('is').input_ids[offset:])
        input_ids.extend(soft_prompt_ids)
        input_ids.append(tokenizer.convert_tokens_to_ids('.'))
        user_prompt = sep_prompt_list[0].split(f'{personalization_token} is <soft token>. ')[1]
        input_ids.extend(tokenizer(user_prompt).input_ids[offset:])
        input_ids.append(image_token_index)
        
        user_ques_ans = sep_prompt_list[1]
        sep_prompt_list = user_ques_ans.split(personalization_token)
        for i, sub_prompt in enumerate(sep_prompt_list):
            while sub_prompt[0] == ' ':
                sub_prompt = sub_prompt[1:]
            while sub_prompt[-1] == ' ':
                sub_prompt = sub_prompt[:-1]
            input_ids.extend(tokenizer(sub_prompt).input_ids[offset:])
            if i < len(sep_prompt_list) - 1:
                input_ids.append(personalization_token_id)
    else:
        sys_prompt = prompt.split(f' {personalization_token} is <soft token>. ')[0]
        prompt_chunks = tokenizer(prompt).input_ids
        offset = 1
        input_ids = []
        input_ids.append(prompt_chunks[0])
        input_ids.extend(tokenizer(sys_prompt).input_ids[offset:])
        input_ids.append(personalization_token_id)
        input_ids.extend(tokenizer('is').input_ids[offset:])
        input_ids.extend(soft_prompt_ids)
        input_ids.append(tokenizer.convert_tokens_to_ids('.'))
        user_prompt = prompt.split(f' {personalization_token} is <soft token>. ')[1]
        sep_prompt_list = user_prompt.split(personalization_token)
        for i, sub_prompt in enumerate(sep_prompt_list):
            while sub_prompt[0] == ' ':
                sub_prompt = sub_prompt[1:]
            while sub_prompt[-1] == ' ':
                sub_prompt = sub_prompt[:-1]
            input_ids.extend(tokenizer(sub_prompt).input_ids[offset:])
            if i < len(sep_prompt_list) - 1:
                input_ids.append(personalization_token_id)
    
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def parse_text_files(txt_file):
    with open(txt_file, "r") as f:
        data = f.read()
    
    return data.split("\n")

def preprocess_v1(
    sources,
    tokenizer,
    personalization_id,
    num_soft_tokens,
    has_image=False
):
    conv = conv_templates["personalized"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer,personalization_id,num_soft_tokens=num_soft_tokens, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer, personalization_id, num_soft_tokens=num_soft_tokens))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer, personalization_id, num_soft_tokens=num_soft_tokens)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_dir", type=str)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--ref_img", type=str)
    parser.add_argument("--tgt_img", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_train_steps", type=int, default=200)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--pos_ques", type=str, default="personalization_dataset/pos_question.txt")
    parser.add_argument("--pos_ans", type=str, default="personalization_dataset/pos_answer.txt")
    parser.add_argument("--neg_ques", type=str, default="personalization_dataset/neg_question.txt")
    parser.add_argument("--neg_ans", default="personalization_dataset/neg_answer.txt")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--importance_weight", type=float, default=1.0)
    parser.add_argument("--other_weights", type=float, default=0.1)
    parser.add_argument("--task", type=str)
    parser.add_argument("--infer_ref_img", type=str)
    parser.add_argument("--infer_query_img", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--pos_prob", type=float, default=0.5)
    parser.add_argument("--data_eval_file", type=str)
    parser.add_argument("--use_features", type=str, choices=["dino", "face"], default="dino")
    parser.add_argument("--face_embedding_dir", type=str)
    parser.add_argument("--yes_no_ratio", type=float, default=0.5)
    parser.add_argument("--question", type=str)
    parser.add_argument("--num_query", type=int, default=16)
    parser.add_argument("--ref_img_list", type=str)
    args = parser.parse_args()
    return args


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_output_dim, lm_head_output_dim):
        super().__init__()
        self.embed_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_output_dim)
        )
        self.lm_head_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lm_head_output_dim)
        )
    
    def forward(self, x):
        return self.embed_projection(x), self.lm_head_projection(x)
    
def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        
        self.linear_weight_mapping = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):

        lm_weight = self.linear_weight_mapping(x[:, 0, :])
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        
        return self.norm_out(latents), lm_weight


def convert_ids_to_tokens(
    self, ids: Union[int, List[int]], skip_special_tokens: bool = False
) -> Union[str, List[str]]:
    """
    Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
    added tokens.

    Args:
        ids (`int` or `List[int]`):
            The token id (or token ids) to convert to tokens.
        skip_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to remove special tokens in the decoding.

    Returns:
        `str` or `List[str]`: The decoded token(s).
    """
    if isinstance(ids, int):
        if ids in self._added_tokens_decoder:
            return self._added_tokens_decoder[ids].content
        else:
            return self._convert_id_to_token(ids)
    tokens = []
    for index in ids:
        index = int(index)
        if skip_special_tokens and index in self.all_special_ids:
            continue
        if index in self._added_tokens_decoder:
            tokens.append(self._added_tokens_decoder[index].content)
        elif index == self.vocab_size:
            tokens.append('<sks>')
        else:
            tokens.append(self._convert_id_to_token(index))
    return tokens

def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens + 16, pad_to_multiple_of)
    if hasattr(old_embeddings, "_hf_hook"):
        hook = old_embeddings._hf_hook
        add_hook_to_module(new_embeddings, hook)
    old_embeddings_requires_grad = old_embeddings.weight.requires_grad
    new_embeddings.requires_grad_(old_embeddings_requires_grad)
    self.set_input_embeddings(new_embeddings)

    # Update new_num_tokens with the actual size of new_embeddings
    if pad_to_multiple_of is not None:
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                new_num_tokens = new_embeddings.weight.shape[0]
        else:
            new_num_tokens = new_embeddings.weight.shape[0]
    
    if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        if hasattr(old_lm_head, "_hf_hook"):
            hook = old_lm_head._hf_hook
            add_hook_to_module(new_lm_head, hook)
        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_embeddings(new_lm_head)
    
    return self.get_input_embeddings()

def _resize_token_embeddings_multiple(self, new_num_tokens, pad_to_multiple_of=None):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens + 32, pad_to_multiple_of)
    if hasattr(old_embeddings, "_hf_hook"):
        hook = old_embeddings._hf_hook
        add_hook_to_module(new_embeddings, hook)
    old_embeddings_requires_grad = old_embeddings.weight.requires_grad
    new_embeddings.requires_grad_(old_embeddings_requires_grad)
    self.set_input_embeddings(new_embeddings)

    # Update new_num_tokens with the actual size of new_embeddings
    if pad_to_multiple_of is not None:
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                new_num_tokens = new_embeddings.weight.shape[0]
        else:
            new_num_tokens = new_embeddings.weight.shape[0]
    
    if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        if hasattr(old_lm_head, "_hf_hook"):
            hook = old_lm_head._hf_hook
            add_hook_to_module(new_lm_head, hook)
        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_embeddings(new_lm_head)
    
    return self.get_input_embeddings()


@torch.no_grad()
def inference(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    tokenizer.convert_ids_to_tokens = types.MethodType(convert_ids_to_tokens, tokenizer)
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    max_seq_length=257
    proj_layer = Resampler(
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=17,
        output_dim=4096,
        ff_mult=4
    )

    vocab_size = tokenizer.vocab_size
    model._resize_token_embeddings = types.MethodType(_resize_token_embeddings, model)
    model.resize_token_embeddings(vocab_size + 1)

    state_dict = torch.load(os.path.join(args.output_dir, args.checkpoint_path))
    proj_layer.load_state_dict(state_dict)
    proj_layer.to("cuda", dtype=torch.float16)
    dino_model.to("cuda", dtype=torch.float16)

    ref_img_dir = args.infer_ref_img
    query_img_dir = args.infer_query_img
    question=args.question
    question = "<image>" + "\n" + question

    conv = conv_templates["personalized"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    ref_img = Image.open(ref_img_dir).convert("RGB")
    query_img = Image.open(query_img_dir)
    image_size = query_img.size
    ref_img = dino_processor(ref_img, return_tensors="pt")["pixel_values"]
    query_img = process_images([query_img], image_processor, model.config)

    query_img = query_img.to("cuda", dtype=torch.float16)

    prompt = conv.get_prompt()
    print(prompt)
    input_ids = tokenizer_image_token(prompt,
                                      tokenizer,
                                      vocab_size)
    input_ids = input_ids.unsqueeze(0)
    ref_img = ref_img.to("cuda", dtype=torch.float16)
    dino_features = dino_model(pixel_values=ref_img).last_hidden_state
    learned_embeddings, proj_weight = proj_layer(dino_features)
    for i in range(learned_embeddings.shape[1]):
        model.model.embed_tokens.weight.data[i+vocab_size] = learned_embeddings[0, i, :]
    model.lm_head.weight.data[vocab_size] = proj_weight[0]

    output_ids = model.generate(
        input_ids,
        images=query_img,
        image_sizes=[image_size],
        do_sample=True if args.temperature > 0 else False,
        # do_sample=False,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_cache=True
    )

    outputs = tokenizer.decode(output_ids[0]).strip()
    print(outputs)

@torch.no_grad()
def conversation(args):
    from llava.mm_utils import tokenizer_image_token
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    tokenizer.convert_ids_to_tokens = types.MethodType(convert_ids_to_tokens, tokenizer)
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    proj_layer = Resampler(
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=17,
        output_dim=4096,
        ff_mult=4
    )
    vocab_size = tokenizer.vocab_size
    model._resize_token_embeddings = types.MethodType(_resize_token_embeddings, model)
    model.resize_token_embeddings(vocab_size + 1)

    state_dict = torch.load(os.path.join(args.output_dir, args.checkpoint_path))
    proj_layer.load_state_dict(state_dict)
    proj_layer.to("cuda", dtype=torch.float16)
    dino_model.to("cuda", dtype=torch.float16)

    conv_mode = "personalized"
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    query_img = load_image(args.infer_query_img)
    image_size = query_img.size
    ref_img = Image.open(args.infer_ref_img).convert("RGB")
    ref_img = dino_processor(ref_img, return_tensors="pt")["pixel_values"]
    ref_img = ref_img.to("cuda", dtype=torch.float16)
    dino_features = dino_model(pixel_values=ref_img).last_hidden_state
    learned_embeddings, proj_weight = proj_layer(dino_features)
    for i in range(learned_embeddings.shape[1]):
        model.model.embed_tokens.weight.data[i+vocab_size] = learned_embeddings[0, i, :]
    model.lm_head.weight.data[vocab_size] = proj_weight[0]
    # Similar operation in model_worker.py
    query_img = process_images([query_img], image_processor, model.config)
    query_img = query_img.to("cuda", dtype=torch.float16)
    image = query_img

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        output_ids = model.generate(
            input_ids,
            images=query_img,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            # do_sample=False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

        breakpoint()

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
args = parse_arguments()
# conversation(args)
inference(args)