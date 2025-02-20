from typing import Optional, Tuple
import torch
from torch import nn
from labml_helpers.module import Module

class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()
        # 创建线性层，将隐藏状态和输入向量映射到四个门的输出
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        # 如果需要应用层归一化
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # 计算 $i_t$, $f_t$, $g_t$ 和 $o_t$ 的线性变换
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim=-1)

        # 应用层归一化
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]
        # 拆分 $i_t$, $f_t$, $g_t$ 和 $o_t$
        i, f, g, o = ifgo

        # 计算 $c_t$
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        # 计算 $h_t$
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next

class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 创建多个 LSTM 层
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # 获取输入张量的时间步长和批次大小
        n_steps, batch_size = x.shape[:2]
        # 如果状态为空，则初始化状态
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            h, c = state

        # 遍历时间步长，更新隐藏状态和记忆状态
        out = []
        for i in range(n_steps):
            inp = x[i]
            for j in range(self.n_layers):
                h[j], c[j] = self.cells[j](inp, h[j], c[j])
                inp = h[j]
            out.append(h[-1])

        # 将输出堆叠为张量
        out = torch.stack(out)
        h = torch.stack(h)
        # c = torch.stack(c)

        return out, h