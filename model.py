"""A small residual policy+value network (AlphaZero/LC0-style).

Small enough to train on CPU: ~1M parameters at default settings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class PolicyHead(nn.Module):
    """1x1 conv producing the 73 action planes per square (4672 logits)."""

    def __init__(self, filters, action_planes=73):
        super().__init__()
        self.conv = nn.Conv2d(filters, action_planes, kernel_size=1)

    def forward(self, x):
        p = self.conv(x)                       # (B, 73, 8, 8)
        p = p.permute(0, 2, 3, 1).contiguous()  # (B, 8, 8, 73)
        return p.view(p.size(0), -1)           # (B, 4672)  index = sq*73 + plane


class ValueHead(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        v = F.relu(self.bn(self.conv(x)))
        v = v.flatten(1)
        v = F.relu(self.fc1(v))
        return torch.tanh(self.fc2(v))


class AlphaZeroNet(nn.Module):
    def __init__(self, planes=18, filters=32, res_blocks=6, action_planes=73):
        super().__init__()
        self.input_conv = nn.Conv2d(planes, filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(res_blocks)])
        self.policy_head = PolicyHead(filters, action_planes)
        self.value_head = ValueHead(filters)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.blocks(x)
        return self.policy_head(x), self.value_head(x)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
