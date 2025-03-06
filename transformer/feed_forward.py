import torch
import torch.nn as nn

from labml_helpers.module import Module

# FFN模块
class FeedForward(Module):

    def __init__(self, d_model: int, d_ff: int,  # 模型维度和前馈网络维度
                 dropout_prob: float = 0.1,  # dropout概率
                 activation: nn.Module = nn.ReLU,  # 激活函数
                 is_gated: bool = False,  # 是否使用门控
                 bias1: bool = True,  # 第一个线性层是否偏置
                 bias2: bool = True,  # 第二个线性层是否偏置
                 bias_gate: bool = True):  # 门控线性层是否偏置
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff, bias=bias1)  # 第一个线性层
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias2)  # 第二个线性层
        self.dropout = nn.Dropout(dropout_prob)  # dropout
        self.activation = activation()  # 激活函数

        self.is_gated = is_gated  # 是否使用门控
        if is_gated:
            self.gate = nn.Linear(d_model, d_ff, bias=bias_gate)  # 门控线性层

    def forward(self, x: torch.Tensor):
        # 第一个线性层
        g = self.linear1(x)
        g = self.activation(g)

        # 如果使用门控
        if self.is_gated:
            # 计算门控
            x = g * self.gate(x)
        else:
            x = g
        # dropout
        x = self.dropout(x)
        # 第二个线性层
        x = self.linear2(x)

        return x