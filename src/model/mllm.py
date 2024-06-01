import torch
import torch.nn as nn
import math
from .model import make_loss, freeze_model
from .huggingface import make_hf_model


class MLLM(nn.Module):
    def __init__(self, data_shape, target_size, llm_model_name, num_hidden_layers, num_data_tokens):
        super().__init__()
        self.data_shape = data_shape
        self.input_size = math.prod(data_shape)
        self.target_size = target_size
        self.llm_model_name = llm_model_name
        self.num_hidden_layers = num_hidden_layers
        self.num_data_tokens = num_data_tokens
        self.llm, self.tokenizer = make_hf_model(llm_model_name, num_hidden_layers)
        freeze_model(self.llm)

        self.input_embedding = self.llm.get_input_embeddings()
        self.hidden_size = self.input_embedding.embedding_dim
        self.encoder = nn.Conv2d(self.data_shape[0], self.hidden_size, 3, 1, 1)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.linear = nn.Linear(self.hidden_size, target_size)

    def feature(self, x):
        x = x.reshape(x.size(0), -1)
        return x

    def output(self, x):
        x = self.linear(x)
        return x

    def f(self, x):
        x = self.feature(x)
        x = self.output(x)
        return x

    def forward(self, input):
        output = {}
        data = input['data']
        encoded = self.encoder(data)
        encoded = encoded.view(encoded.size(0), encoded.size(1), -1).permute(0, 2, 1)

        input_ids = input['input_ids']
        prompt_embeddings = self.input_embedding(input_ids)

        encoded = torch.cat([encoded, prompt_embeddings], dim=1)

        decoded = self.llm(inputs_embeds=encoded).last_hidden_state

        output['target'] = self.linear(decoded[:, -1])
        output['loss'] = make_loss(output, input)
        return output


def mllm(cfg):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    llm_model_name = cfg['llm_model_name']
    num_hidden_layers = cfg['mllm']['num_hidden_layers']
    num_data_tokens = cfg['mllm']['num_data_tokens']
    model = MLLM(data_shape, target_size, llm_model_name, num_hidden_layers, num_data_tokens)
    return model
