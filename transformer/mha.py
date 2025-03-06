import math
from typing import Optional, List

import torch
import torch.nn as nn

from labml import tracker

# 准备多头注意力
class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        self.linear = nn.Linear(d_model, d_k * heads, bias=bias)  # 线性变换
        self.d_k = d_k  # 每个头的维度
        self.heads = heads  # 头的数量

    def forward(self, x: torch.Tensor):
        # 对最后一维进行线性变换，并将其分为多个头
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)

        return x

# 多头注意力模块
class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout_prob : float = 0.1, bias: bool = True):
        super().__init__()

        self.heads = heads  # 头的数量
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)  # 查询
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)  # 键
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)  # 值

        self.softmax = nn.Softmax(dim=-1)  # 在键的时间维度上进行softmax
        self.output = nn.Linear(d_model, d_model)  # 输出线性变换
        self.dropout = nn.Dropout(dropout_prob)  # dropout

        self.scale = 1 / math.sqrt(self.d_k)  # 缩放因子
        self.attn =None  # 存储注意力信息

    # 计算注意力得分
    def get_score(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->bhij', query, key)
    
    # 准备掩码
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        # 检查掩码形状是否正确，mask形状可以是[1或seq_len_q, seq_len_k, 1或batch_size]
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)  # 所有头部使用相同掩码
        return mask
    
    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):

        seq_len_q, batch_size, _ = query.shape  # 获取查询序列长度、批次大小

        # 准备查询、键和值
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 准备掩码向量
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # 计算注意力得分
        scores = self.get_score(query, key) * self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn = self.softmax(scores)
        # 调试时保存注意力信息
        tracker.debug('attn', attn)
        # 应用dropout
        attn = self.dropout(attn)
        # 加权求和，计算注意力得分
        x = torch.einsum('bhij,bjhd->bihd', attn, value)
        
        self.attn = attn.detach()  # 保存注意力信息

        # 连接多个头
        x = x.view(seq_len_q, batch_size, self.heads * self.d_k)
        # 输出线性变换
        x = self.output(x)

        return x