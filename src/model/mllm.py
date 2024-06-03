import torch
import torch.nn as nn
import math
from .model import make_loss, freeze_model
from .huggingface import make_hf_model
from .embed import PatchEmbedding

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class MLLM(nn.Module):
    def __init__(self, data_shape, target_size, patch_size, stride, llm_model_name, num_hidden_layers, num_data_tokens):
        super().__init__()
        self.data_shape = data_shape
        self.input_size = math.prod(data_shape)
        self.target_size = target_size
        self.patch_size = patch_size
        self.stride = stride
        self.llm_model_name = llm_model_name
        self.num_hidden_layers = num_hidden_layers
        self.num_data_tokens = num_data_tokens
        self.llm, self.tokenizer = make_hf_model(llm_model_name, num_hidden_layers)
        freeze_model(self.llm)

        self.word_embeddings = self.llm.get_input_embeddings()
        self.vocab_size = self.word_embeddings.num_embeddings
        self.hidden_size = self.word_embeddings.embedding_dim
        self.patch_hidden_size = 32
        self.num_heads = 2
        self.patch_embedding = PatchEmbedding(self.patch_hidden_size, self.patch_size, self.stride, False)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_data_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.patch_hidden_size, self.num_heads, self.hidden_size)
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
        patched = self.patch_embedding(data)
        source_embeddings = self.mapping_layer(self.word_embeddings.weight.permute(1, 0)).permute(1, 0)
        encoded = self.reprogramming_layer(patched, source_embeddings, source_embeddings)

        input_ids = input['input_ids']
        prompt_embeddings = self.word_embeddings(input_ids)

        encoded = torch.cat([encoded, prompt_embeddings], dim=1)

        decoded = self.llm(inputs_embeds=encoded).last_hidden_state
        output['target'] = self.linear(decoded.mean(dim=1))
        output['loss'] = make_loss(output, input)
        return output


def mllm(cfg):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    patch_size = cfg['mllm']['patch_size']
    stride = cfg['mllm']['stride']
    llm_model_name = cfg['mllm']['llm_model_name']
    num_hidden_layers = cfg['mllm']['num_hidden_layers']
    num_data_tokens = cfg['mllm']['num_data_tokens']
    model = MLLM(data_shape, target_size, patch_size, stride, llm_model_name, num_hidden_layers, num_data_tokens)
    return model
