import torch
import torch.nn as nn


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, hidden_size, patch_size, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_size = patch_size
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_size, hidden_size)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(hidden_size)

        # Residual dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # do patching
        # print(x.size())
        x = x.view(x.size(0), x.size(1), -1)
        x = self.padding_patch_layer(x)
        # print(x.size())
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # print(x.size())
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # print(x.size())
        # Input encoding
        x = self.value_embedding(x)
        # print(x.size())
        if self.dropout is not None:
            x = self.dropout(x)
        return x
