import torch
import torch.nn as nn
import math
from .model import init_param, make_loss


class MLLM(nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        input_size = math.prod(data_shape)
        self.linear = nn.Linear(input_size, target_size)

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
        x = input['data']
        x = self.f(x)
        output['target'] = x
        output['loss'] = make_loss(output, input)
        return output


def mllm(cfg):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    model = MLLM(data_shape, target_size)
    return model