import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

# For temporal attention
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size+1
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, C, H, L, E = queries.shape 
        scale = self.scale or 1. / sqrt(E)

        scores = queries @ keys.transpose(-1,-2)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores 

        if sigma is not None: 
            window_size = attn.shape[-1] # except CLS
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, 1, window_size)  # B C H L L

            prior = self.distances.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], sigma.shape[2], 1, 1).cuda()
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        else:
            prior= None

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = series @ values

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, C, _ = queries.shape
        _, S, C,  _ = keys.shape
        H = self.n_heads
        x = queries

        queries = self.query_projection(queries).view(B, L, C, H, -1).permute(0,2,3,1,4) 
        keys = self.key_projection(keys).view(B, L, C, H, -1).permute(0,2,3,1,4)
        values = self.value_projection(values).view(B, L, C, H, -1).permute(0,2,3,1,4)
        sigma = self.sigma_projection(x).view(B, L, C, H).permute(0,2,3,1) 

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, C, -1)

        return self.out_projection(out), series, prior, sigma

#------------------------------------------------------------------

# For channel attention
class AnomalyAttention_new(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention_new, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, queries, keys, values, attn_mask):
        B, C, H, L, E = queries.shape 
        scale = self.scale or 1. / sqrt(E)

        scores = queries @ keys.transpose(-1,-2)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores 

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = series @ values

        return V.contiguous()


class AttentionLayer_new(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer_new, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, C, _ = queries.shape
        _, S, C,  _ = keys.shape
        H = self.n_heads
        x = queries

        queries = self.query_projection(queries).view(B, L, C, H, -1).permute(0,2,3,1,4) # [bs,100,25,dim] -> [bs, 100(len), 25(C), head(8), 64(dim/H)]
        keys = self.key_projection(keys).view(B, L, C, H, -1).permute(0,2,3,1,4)
        values = self.value_projection(values).view(B, L, C, H, -1).permute(0,2,3,1,4)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, C, -1)

        return self.out_projection(out)
