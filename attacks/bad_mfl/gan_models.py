from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int=4, base_channels: int=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 3, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_tensor, mask_tensor], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = F.interpolate(b, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d3, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out(d2)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int=3, base_channels: int=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(image_tensor)


def pil_to_tensor(image: Image.Image, device: torch.device, image_size: int=224) -> torch.Tensor:
    image = image.convert('RGB').resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_pil(image_tensor: torch.Tensor, size: tuple[int, int] | None=None) -> Image.Image:
    tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)[0]
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    image = Image.fromarray(arr)
    if size is not None and image.size != size:
        image = image.resize(size)
    return image


def box_to_mask(box, image_size: tuple[int, int], device: torch.device, resize_to: int=224) -> torch.Tensor:
    width, height = image_size
    x0, y0, x1, y1 = [int(v) for v in box]
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    mask[:, :, y0:y1, x0:x1] = 1.0
    if (height, width) != (resize_to, resize_to):
        mask = F.interpolate(mask, size=(resize_to, resize_to), mode='nearest')
    return mask
