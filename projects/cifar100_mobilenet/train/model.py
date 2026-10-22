"""
model.py: MobileNet Architecture adapted for CIFAR-100 (32x32 resolution)
"""

import struct
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Depthwise 3x3
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU6(inplace=True)

        # Pointwise 1x1
        self.pw_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        return x


class MobileNetCIFAR100(nn.Module):
    """
    MobileNet adapted for CIFAR-100 (32x32 images).
    Stride of initial conv is 1 to preserve spatial resolution.
    Downsampling occurs via depthwise separable blocks (32 -> 16 -> 8 -> 4).
    """
    def __init__(self, num_classes: int = 100):
        super().__init__()
        # Stem: 3 -> 32, 32x32
        self.stem_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU6(inplace=True)

        # Depthwise Separable Blocks
        # (in_c, out_c, stride)
        cfg = [
            (32, 64, 1),    # Block 0: 32x32
            (64, 128, 2),   # Block 1: 16x16
            (128, 128, 1),  # Block 2: 16x16
            (128, 256, 2),  # Block 3: 8x8
            (256, 256, 1),  # Block 4: 8x8
            (256, 512, 2),  # Block 5: 4x4
            (512, 512, 1),  # Block 6: 4x4
        ]
        blocks = []
        for in_c, out_c, s in cfg:
            blocks.append(DepthwiseSeparableConv(in_c, out_c, stride=s))
        self.blocks = nn.ModuleList(blocks)

        # Head: GlobalAvgPool + Linear
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def fuse_conv_bn_weights(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fuses Conv2d and BatchNorm2d weights into equivalent standalone Conv2d weight and bias.
    Matches rcre_nn_fuse_conv_batchnorm exactly.
    """
    w = conv.weight
    mean = bn.running_mean
    var_val = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    inv_std = 1.0 / torch.sqrt(var_val + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale

    if conv.groups > 1: # Depthwise
        # w shape: [C, 1, K, K], scale shape: [C]
        w_fused = w * scale.view(-1, 1, 1, 1)
    else:
        # w shape: [C_out, C_in, K, K], scale shape: [C_out]
        w_fused = w * scale.view(-1, 1, 1, 1)

    b_fused = shift
    return w_fused.detach().cpu().numpy(), b_fused.detach().cpu().numpy()


def export_mobilenet_to_rcre_binary(model: MobileNetCIFAR100, output_path: str):
    """
    Exports a trained MobileNet model to RCRE binary format (RCRE_MOB) with fused BN layers.
    """
    model.eval()
    tensors_to_write = []

    # 1. Stem Conv + BN
    w_stem, b_stem = fuse_conv_bn_weights(model.stem_conv, model.stem_bn)
    tensors_to_write.append(("stem_conv_w", w_stem))
    tensors_to_write.append(("stem_conv_b", b_stem))

    # 2. Blocks
    for i, block in enumerate(model.blocks):
        w_dw, b_dw = fuse_conv_bn_weights(block.dw_conv, block.dw_bn)
        tensors_to_write.append((f"block_{i}_dw_w", w_dw))
        tensors_to_write.append((f"block_{i}_dw_b", b_dw))

        w_pw, b_pw = fuse_conv_bn_weights(block.pw_conv, block.pw_bn)
        tensors_to_write.append((f"block_{i}_pw_w", w_pw))
        tensors_to_write.append((f"block_{i}_pw_b", b_pw))

    # 3. FC Linear
    w_fc = model.fc.weight.detach().cpu().numpy()
    b_fc = model.fc.bias.detach().cpu().numpy()
    tensors_to_write.append(("fc_w", w_fc))
    tensors_to_write.append(("fc_b", b_fc))

    # Write binary file
    # Header:
    # Magic: "RCRE_MOB" (8 bytes)
    # Version: uint32 (1)
    # Num tensors: uint32
    with open(output_path, "wb") as f:
        magic = b"RCRE_MOB"
        version = 1
        num_tensors = len(tensors_to_write)
        f.write(magic)
        f.write(struct.pack("<II", version, num_tensors))

        for name, arr in tensors_to_write:
            ndim = arr.ndim
            shape = list(arr.shape) + [1] * (4 - ndim)
            f.write(struct.pack("<I", ndim))
            f.write(struct.pack("<IIII", *shape[:4]))
            # Write raw float32 bytes
            f.write(arr.astype("float32").tobytes())

    print(f"✅ Successfully exported {len(tensors_to_write)} tensors to {output_path} (RCRE_MOB v1)")
