import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)+1] # CLS


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.c_in = c_in
        self.value_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.cls_tokens = nn.Parameter(torch.randn(1,1,c_in,d_model)) # [1,1,C,D]


    def forward(self, x):

        enc_out = [None] * self.c_in
        for ch in range(self.c_in):
            enc_out[ch] = self.value_embedding(x[:,:,ch].unsqueeze(2)).unsqueeze(0)
        val_emb = torch.cat(enc_out).permute(1,2,0,3) # [B,L,C,D]

        #- CLS token
        cls_tokens = self.cls_tokens.repeat(val_emb.shape[0],1,1,1) # [B,1,C,D]
        val_emb = torch.cat([cls_tokens, val_emb], dim=1) # [B,L+1,C,D]

        pos_emb = self.position_embedding(x).unsqueeze(2).repeat(1,1,self.c_in,1)
        x = val_emb + pos_emb

        return self.dropout(x)  # [B,L+1,C,D]
