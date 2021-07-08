import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
from einops import rearrange, repeat
from functools import partial

import torchkeras


import torch.nn as nn
import math

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]



class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.1, scale=True):
        """
        :param dim:
        :param n_head: multi-head
        :param scale: 是否scale输出
        """
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, head_dim * heads)
        self.to_k = nn.Linear(dim, head_dim * heads)
        self.to_v = nn.Linear(dim, head_dim * heads)
        self.to_out = nn.Linear(head_dim * heads, dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

        if scale:
            self.scale = 1 / math.sqrt(head_dim)
        else:
            self.scale = 1

    def forward(self, x, context = None):
        x = self.layer_norm(x)
        b, n, _, h = *x.shape, self.heads

        cross_attend = exists(context)
        context = default(context, x)
        
        batch_size, max_len, dim = x.size()

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h i d -> b i (h d)', h = h)
        out = self.to_out(out)
        # return self.layer_norm(out + x)
        return out + x


class MH_DEAttn(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.1, scale=True):
        """
        :param dim:
        :param n_head: multi-head
        :param scale: 是否scale输出
        """
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, head_dim * heads)
        self.to_k = nn.Linear(dim, head_dim * heads)
        self.to_v = nn.Linear(dim, head_dim * heads)
        self.to_out = nn.Linear(head_dim * heads + dim, dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

        if scale:
            self.scale = 1 / math.sqrt(head_dim)
        else:
            self.scale = 1

    def forward(self, x, x0):
        x = self.layer_norm(x)
        b, n, _, h = *x.shape, self.heads
        
        batch_size, max_len, dim = x.size()

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h i d -> b i (h d)', h = h)
        out = torch.cat([out, x0], dim=2)
        out = self.to_out(out)
        # return self.layer_norm(out + x)
        return out + x


if __name__ == '__main__':
    model = MH_DEAttn(dim=64, heads=4, head_dim=8)
    x = torch.randn(1, 100, 64)
    x0 = torch.randn(1, 100, 64)
    print(model(x, x0).shape)