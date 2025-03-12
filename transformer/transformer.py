import math

import torch
import torch.nn as nn

from labml_nn.utils import clone_module_list
from feed_forward import FeedForward
from mha import MultiHeadAttention
from positional_encoding import get_positional_encoding

# 嵌入token并添加固定位置编码
# 这个类将输入的token映射为向量并添加预计算的位置编码
class EmbeddingsWithPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, n_vocab: int, max_len: int =5000):
        """
        初始化嵌入层和位置编码
        
        参数:
            d_model: 模型的维度/嵌入维度
            n_vocab: 词汇表大小
            max_len: 支持的最大序列长度
        """
        super().__init__()

        self.linear = nn.Embedding(n_vocab, d_model)  # 将token ID转换为向量表示
        self.d_model = d_model
        # 注册固定的位置编码作为buffer(不参与优化)
        self.register_buffer('positional_encoding', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        """
        将输入token转换为嵌入并添加位置编码
        
        参数:
            x: 形状为[seq_len, batch_size]的输入token ID
            
        返回:
            形状为[seq_len, batch_size, d_model]的嵌入向量
        """
        pe = self.positional_encoding[:x.shape[0]].detach().requires_grad_(False)
        x = self.linear(x) * math.sqrt(self.d_model) + pe  # 缩放嵌入并加上位置编码
        return x
        
# 嵌入token并添加参数化位置编码
# 与固定位置编码不同，这个类使用可学习的位置编码
class EmbeddingWithLearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, n_vocab: int, max_len: int =5000):
        """
        初始化嵌入层和可学习位置编码
        
        参数:
            d_model: 模型的维度/嵌入维度
            n_vocab: 词汇表大小
            max_len: 支持的最大序列长度
        """
        super().__init__()

        self.linear = nn.Embedding(n_vocab, d_model)  # 将token ID转换为向量表示
        self.d_model = d_model
        # 创建可学习的位置编码参数
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        将输入token转换为嵌入并添加可学习位置编码
        
        参数:
            x: 形状为[seq_len, batch_size]的输入token ID
            
        返回:
            形状为[seq_len, batch_size, d_model]的嵌入向量
        """
        pe = self.positional_encoding[:x.shape[0]]
        x = self.linear(x) * math.sqrt(self.d_model) + pe  # 缩放嵌入并加上位置编码
        return x

# Transformer Layer
# Transformer的基本构建块，包含自注意力、交叉注意力和前馈网络
class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        初始化Transformer层
        
        参数:
            d_model: 模型的维度
            self_attn: 自注意力模块
            src_attn: 用于编码器-解码器注意力的多头注意力模块，在编码器层中为None
            feed_forward: 前馈网络模块
            dropout_prob: dropout概率
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn  # 自注意力层
        self.src_attn = src_attn    # 编码器-解码器注意力层（仅解码器使用）
        self.feed_forward = feed_forward  # 前馈网络
        self.dropout = nn.Dropout(dropout_prob)
        # 层归一化
        self.norm_self_attn = nn.LayerNorm(d_model)
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.is_save_ff_input = False  # 是否保存前馈网络的输入（用于分析）

    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        """
        Transformer层的前向传播
        
        参数:
            x: 输入张量 [seq_len, batch_size, d_model]
            mask: 自注意力的掩码
            src: 编码器输出（用于解码器） [seq_len, batch_size, d_model]
            src_mask: 编码器-解码器注意力的掩码
            
        返回:
            处理后的张量 [seq_len, batch_size, d_model]
        """
        # 第一个子层: 自注意力（Pre-LN架构）
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)  # 残差连接
        
        # 第二个子层: 编码器-解码器注意力（仅解码器中使用）
        if src is not None:
            z = self.norm_src_attn(x)
            src_attn = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(src_attn)  # 残差连接

        # 第三个子层: 前馈网络
        z = self.norm_ff(x)
        if self.is_save_ff_input:
            self.ff_input = z.clone()  # 保存前馈网络输入（用于分析）

        ff = self.feed_forward(z)
        x = x + self.dropout(ff)  # 残差连接

        return x

# 编码器由多个Transformer层组成
class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        """
        初始化编码器
        
        参数:
            layer: 单个Transformer编码器层
            n_layers: 层数
        """
        super().__init__()

        # 创建n_layers个相同的层
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)  # 最终的层归一化

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        编码器的前向传播
        
        参数:
            x: 输入嵌入 [seq_len, batch_size, d_model]
            mask: 注意力掩码
            
        返回:
            编码后的表示 [seq_len, batch_size, d_model]
        """
        # 通过每一层
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        # 最终的层归一化
        return self.norm(x)
    
# 解码器由多个Transformer层组成
class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        """
        初始化解码器
        
        参数:
            layer: 单个Transformer解码器层
            n_layers: 层数
        """
        super().__init__()

        # 创建n_layers个相同的层
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)  # 最终的层归一化

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """
        解码器的前向传播
        
        参数:
            x: 目标序列的嵌入 [tgt_seq_len, batch_size, d_model]
            memory: 编码器的输出 [src_seq_len, batch_size, d_model]
            src_mask: 源序列的掩码
            tgt_mask: 目标序列的掩码（通常包含注意力掩码以防止关注未来位置）
            
        返回:
            解码后的表示 [tgt_seq_len, batch_size, d_model]
        """
        # 通过每一层，每层都使用编码器的输出memory
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # 最终的层归一化
        return self.norm(x)
    
# 输出生成器，将解码器的输出映射到词汇表大小的分布
class Generator(nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        """
        初始化生成器
        
        参数:
            n_vocab: 词汇表大小
            d_model: 模型维度
        """
        super().__init__()

        # 线性投影到词汇表大小
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x: torch.Tensor):
        """
        生成词汇分布
        
        参数:
            x: 解码器输出 [seq_len, batch_size, d_model]
            
        返回:
            词汇分布对数 [seq_len, batch_size, n_vocab]
        """
        return self.projection(x)
    
# 完整的Transformer编码器-解码器模型
class EncoderDecoder(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        """
        初始化Transformer编码器-解码器
        
        参数:
            encoder: 编码器模块
            decoder: 解码器模块
            src_embed: 源序列的嵌入层
            tgt_embed: 目标序列的嵌入层
            generator: 输出生成器
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # 使用Xavier均匀初始化权重
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        编码源序列
        
        参数:
            src: 源序列token ID [src_seq_len, batch_size]
            src_mask: 源序列掩码
            
        返回:
            编码后的表示 [src_seq_len, batch_size, d_model]
        """
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        解码目标序列
        
        参数:
            memory: 编码器输出 [src_seq_len, batch_size, d_model]
            src_mask: 源序列掩码
            tgt: 目标序列token ID [tgt_seq_len, batch_size]
            tgt_mask: 目标序列掩码
            
        返回:
            解码后的表示 [tgt_seq_len, batch_size, d_model]
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Transformer的完整前向传播
        
        参数:
            src: 源序列token ID [src_seq_len, batch_size]
            tgt: 目标序列token ID [tgt_seq_len, batch_size]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        返回:
            解码后的表示 [tgt_seq_len, batch_size, d_model]
        """
        enc = self.encode(src, src_mask)

        return self.decode(enc, src_mask, tgt, tgt_mask)