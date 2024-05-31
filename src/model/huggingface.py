import os
import torch
import torch.nn as nn
from config import cfg
from transformers import AutoConfig, AutoTokenizer, AutoModel, \
    LlamaConfig, LlamaTokenizer, LlamaModel, \
    GPT2Config, GPT2Tokenizer, GPT2Model, \
    BertConfig, BertTokenizer, BertModel


def make_hf_model(model_name, num_hidden_layers):
    def process_config(hf_config):
        hf_config.num_hidden_layers = num_hidden_layers
        hf_config.output_attentions = True
        hf_config.output_hidden_states = True
        return hf_config

    if 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'bloom' in model_name:
        cfg['model_name_or_path'] = 'bigscience/{}'.format(model_name)
    elif 'bart' in model_name:
        cfg['model_name_or_path'] = 'facebook/{}'.format(model_name)
    elif 'roberta' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
    elif 'gpt' in model_name:
        cfg['model_name_or_path'] = 'openai-community/{}'.format(model_name)
    elif 't5' in model_name:
        cfg['model_name_or_path'] = '{}'.format(model_name)
    elif 'llama-2' in model_name:
        # https://huggingface.co/docs/transformers/main/model_doc/llama2
        # FOLLOW the instruction to run the script: python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir output/llama-2-7b
        cfg['model_name_or_path'] = 'output/{}'.format(model_name)
    elif 'llama' in model_name:
        cfg['model_name_or_path'] = 'huggyllama/{}'.format(model_name)
    elif 'bert' in model_name:
        cfg['model_name_or_path'] = 'google-bert/{}'.format(model_name)
    else:
        raise ValueError('Not valid model name')
    cfg['cache_model_path'] = os.path.join('output', 'model', model_name)
    cfg['cache_tokenizer_path'] = os.path.join('output', 'tokenizer', model_name)

    if any(k in cfg['model_name_or_path'] for k in ("gpt", "opt", "bloom", "llama")):
        padding_side = "left"
    else:
        padding_side = "right"

    if 'llama' in model_name:
        hf_config = LlamaConfig.from_pretrained(cfg['model_name_or_path'])
        hf_config = process_config(hf_config)
        tokenizer = LlamaTokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                   padding_side=padding_side)
        model = LlamaModel.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                           config=hf_config)
    elif 'gpt' in model_name:
        hf_config = GPT2Config.from_pretrained(cfg['model_name_or_path'])
        hf_config = process_config(hf_config)
        tokenizer = GPT2Tokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
        model = GPT2Model.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                          config=hf_config)
    elif 'bert' in model_name:
        hf_config = BertConfig.from_pretrained(cfg['model_name_or_path'])
        hf_config = process_config(hf_config)
        tokenizer = BertTokenizer.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
        model = BertModel.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                          config=hf_config)
    else:
        hf_config = AutoConfig.from_pretrained(cfg['model_name_or_path'])
        hf_config = process_config(hf_config)
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name_or_path'], cache_dir=cfg['cache_tokenizer_path'],
                                                  padding_side=padding_side)
        model = AutoModel.from_pretrained(cfg['model_name_or_path'], cache_dir=cfg['cache_model_path'],
                                          trust_remote_code=True, config=hf_config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
