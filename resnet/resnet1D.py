from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from labml_helpers.module import Module

# 用于快捷连接的线性投影层
class ShortcutProjection(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        return x

# 残差块
class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 第一层卷积、批量归一化、激活函数
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU()
        # 第二层卷积、批量归一化、快捷连接、激活函数
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 当输出维度与输入维度不等时，需要进行快捷连接，否则进行恒等映射
        if in_channels != out_channels or stride != 1:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # 第一层卷积、批量归一化、激活函数
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # 第二层卷积、批量归一化、快捷连接、激活函数
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.act2(x)
        return x

# 瓶颈残差块
class BottleneckResidualBlock(Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super().__init__()
        # 第一层1×1卷积层
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        self.act1 = nn.ReLU()
        # 第二层3×3卷积层
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.act2 = nn.ReLU()
        # 第三层1×1卷积层
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        # 当输出维度与输入维度不等时，需要进行快捷连接，否则进行恒等映射
        if in_channels!= out_channels or stride!= 1:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # 第一层1×1卷积层
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # 第二层3×3卷积层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # 第三层1×1卷积层
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + shortcut
        x = self.act3(x)
        return x

# ResNet模型
class ResNetBase(Module):
    def __init__(self, n_blocks: List[int], n_channels: List[int], bottlencks: Optional[List[int]] = None, img_channels: int = 3, first_keneral_size: int =7):
        super().__init__()

        # 检查输入参数的有效性
        assert len(n_blocks) == len(n_channels)
        assert bottlencks is None or len(n_blocks) == len(bottlencks)

        # 第一层卷积和批量归一化
        self.conv1 = nn.Conv1d(img_channels, n_channels[0], kernel_size=first_keneral_size, stride=1, padding=first_keneral_size // 2)
        self.bn1 = nn.BatchNorm1d(n_channels[0])

        # 初始化残差块列表
        blocks = []
        prev_channels = n_channels[0]

        # 遍历每个阶段构建残差块
        for i, channels in enumerate(n_channels):
            # 第一个块的步长为2，其余为1
            stride = 1

            # 根据是否使用瓶颈结构选择不同的残差块
            if bottlencks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlencks[i], channels, stride))

            # 更新前一层的通道数
            prev_channels = channels

            # 添加剩余的残差块
            for _ in range(n_blocks[i] - 1):
                if bottlencks is None:
                    blocks.append(ResidualBlock(channels, channels, 1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlencks[i], channels, 1))

        # 将所有残差块组合成序列
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.blocks(x)
        return x