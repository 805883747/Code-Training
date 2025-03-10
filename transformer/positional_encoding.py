import math
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_len: int = 5000):
        """
        初始化位置编码模块
        
        参数:
            d_model: 模型的维度，也是词嵌入的维度
            dropout_prob: Dropout概率，用于防止过拟合
            max_len: 支持的最大序列长度
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        # 注册位置编码缓冲区，这些编码不会作为模型参数进行训练
        self.register_buffer('positional_encoding', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数
        
        参数:
            x: 输入张量，形状为[seq_len, batch_size, d_model]
            
        返回:
            添加了位置编码并应用dropout的张量
        """
        # 提取与输入序列长度匹配的位置编码
        pe = self.positional_encoding[:x.shape[0]].detach().requires_grad_(False)
        # 将位置编码添加到输入中
        x = x + pe
        # 应用dropout并返回
        x = self.dropout(x)
        return x

def get_positional_encoding(d_model: int, max_len :int =5000):
    """
    生成位置编码
    
    使用正弦和余弦函数的定义:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    
    参数:
        d_model: 模型的维度
        max_len: 最大序列长度
        
    返回:
        形状为[max_len, 1, d_model]的位置编码张量
    """

    encodings = torch.zeros(max_len, d_model)
    # 创建一个表示位置的列向量 [max_len, 1]
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # 计算分母中2i项的值 
    two_i = torch.arange(0, d_model, step=2, dtype=torch.float32)
    # 计算10000^(2i/d_model)项
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # 使用sin函数计算偶数索引位置的编码
    encodings[:, 0::2] = torch.sin(position * div_term)
    # 使用cos函数计算奇数索引位置的编码
    encodings[:, 1::2] = torch.cos(position * div_term)

    # 扩展维度以匹配预期的输出形状 [max_len, 1, d_model]，并设置为不需要梯度
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings

def _test_positional_encoding():
    """
    测试位置编码并可视化结果
    
    绘制位置编码的几个维度，以展示不同频率的正弦波
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = get_positional_encoding(20, 100)
    # 绘制位置编码的第4-7维
    plt.plot(np.arange(100), pe[:, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.show()

if __name__ == "__main__":
    _test_positional_encoding()