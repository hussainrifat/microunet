# models/unet.py
# Implementation of a small U-Net with <0.1M parameters
# Based on Kist & Döllinger, IEEE Access 2020

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise Conv + Pointwise Conv.
    Much cheaper than regular Conv2d — fewer parameters for same receptive field.
    
    Regular Conv2d(32, 64, 3x3) = 32 * 64 * 3 * 3 = 18,432 parameters
    SeparableConv2d(32, 64)     = 32*9 + 32*64   = 288 + 2048 = 2,336 parameters
    ~8x fewer parameters!
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise: one filter per input channel, captures spatial patterns
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels,   # groups=in_channels means each channel is convolved separately
            bias=False
        )
        # Pointwise: 1x1 conv to mix channels together
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)   # spatial filtering
        x = self.pointwise(x)   # channel mixing
        return x


class ConvBlock(nn.Module):
    """
    One block = two separable convolutions, each followed by BatchNorm and ReLU.
    This is the basic building block used in every encoder and decoder layer.
    
    BatchNorm: normalizes activations to stabilize and speed up training
    ReLU: activation function — sets negative values to zero, adds non-linearity
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # First conv: in_channels -> out_channels
            SeparableConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),  # normalize across batch
            nn.ReLU(inplace=True),         # activation (inplace saves memory)

            # Second conv: out_channels -> out_channels
            SeparableConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MicroUNet(nn.Module):
    """
    A 4-layer U-Net with separable convolutions and skip connections.
    Target: <0.1M parameters, ~0.87 IoU on BAGLS (from reference paper).
    
    Architecture:
    Encoder: 4 layers that downsample the image (make it smaller, more abstract)
    Bottleneck: deepest layer with most channels
    Decoder: 4 layers that upsample back to original size
    Skip connections: concatenate encoder features to decoder (the 'U' shape)
    """
    def __init__(self, in_channels=3, out_channels=1, base_filters=8):
        """
        in_channels: 3 for RGB images
        out_channels: 1 for binary segmentation mask
        base_filters: number of filters in first layer (we use 8, matching paper)
        """
        super().__init__()

        f = base_filters  # shorthand, f=8

        # --- ENCODER (downsampling path) ---
        # Each encoder block doubles the number of filters
        self.enc1 = ConvBlock(in_channels, f)      # 3  -> 8  filters
        self.enc2 = ConvBlock(f,           f * 2)  # 8  -> 16 filters
        self.enc3 = ConvBlock(f * 2,       f * 4)  # 16 -> 32 filters
        self.enc4 = ConvBlock(f * 4,       f * 8)  # 32 -> 64 filters

        # MaxPool: halves spatial dimensions (256->128->64->32->16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK (deepest point of the U) ---
        self.bottleneck = ConvBlock(f * 8, f * 16)  # 64 -> 128 filters

        # --- DECODER (upsampling path) ---
        # Upsample: doubles spatial dimensions back up
        # Each decoder block takes upsampled features + skip connection from encoder
        # That's why in_channels is doubled (f*16 + f*8 = f*24, etc.)
        self.up4    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4   = ConvBlock(f * 16 + f * 8,  f * 8)   # 128+64=192 -> 64

        self.up3    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3   = ConvBlock(f * 8  + f * 4,  f * 4)   # 64+32=96   -> 32

        self.up2    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2   = ConvBlock(f * 4  + f * 2,  f * 2)   # 32+16=48   -> 16

        self.up1    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1   = ConvBlock(f * 2  + f,       f)       # 16+8=24    -> 8

        # --- OUTPUT ---
        # 1x1 conv to map 8 filters -> 1 output channel (the segmentation mask)
        # Sigmoid squashes output to [0,1] — probability of being foreground
        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass — how data flows through the network.
        x shape: [batch, 3, 256, 256]
        """
        # --- ENCODER ---
        e1 = self.enc1(x)           # [B, 8,  256, 256]
        e2 = self.enc2(self.pool(e1))  # [B, 16, 128, 128]
        e3 = self.enc3(self.pool(e2))  # [B, 32, 64,  64 ]
        e4 = self.enc4(self.pool(e3))  # [B, 64, 32,  32 ]

        # --- BOTTLENECK ---
        b  = self.bottleneck(self.pool(e4))  # [B, 128, 16, 16]

        # --- DECODER with skip connections ---
        # torch.cat concatenates along the channel dimension
        # This is the skip connection — encoder features are passed directly to decoder
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # [B, 64, 32,  32 ]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # [B, 32, 64,  64 ]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 16, 128, 128]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 8,  256, 256]

        # --- OUTPUT ---
        out = self.sigmoid(self.output_conv(d1))  # [B, 1, 256, 256]
        return out


def count_parameters(model):
    """Counts total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)