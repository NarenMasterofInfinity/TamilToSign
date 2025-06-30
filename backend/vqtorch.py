import os
import csv
import torch
from vector_quantize_pytorch import VectorQuantize
import torch.nn as nn
import torch.nn.functional as F

class ResConv1DBlock(nn.Module):
    def __init__(self, channels, dilation=1, padding=0):
        super().__init__()
        self.activation1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            stride=1, padding=padding, dilation=dilation
        )
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = self.activation1(x)
        out = self.conv1(out)
        out = self.activation2(out)
        out = self.conv2(out)
        return x + out


class VQVAEEncoder(nn.Module):
    def __init__(self, in_joints=28, in_dims=3, hidden_dim=512):
        super().__init__()
        self.in_channels = in_joints * in_dims
        self.conv_in = nn.Conv1d(self.in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.Sequential(
                    ResConv1DBlock(hidden_dim, dilation=9, padding=9),
                    ResConv1DBlock(hidden_dim, dilation=3, padding=3),
                    ResConv1DBlock(hidden_dim, dilation=1, padding=1),
                )
            )
            self.blocks.append(block)

    def forward(self, x):
        B, T, J, D = x.shape
        x = x.view(B, T, J * D).transpose(1, 2)  # (B, C, T)
        out = self.relu(self.conv_in(x))
        for block in self.blocks:
            out = block(out)
        return out  # (B, hidden_dim, T_down)


class VQVAEDecoder(nn.Module):
    def __init__(self, out_joints=28, out_dims=3, hidden_dim=512):
        super().__init__()
        self.out_channels = out_joints * out_dims
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Sequential(
                nn.Sequential(
                    ResConv1DBlock(hidden_dim, dilation=1, padding=1),
                    ResConv1DBlock(hidden_dim, dilation=3, padding=3),
                    ResConv1DBlock(hidden_dim, dilation=9, padding=9),
                ),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
            )
            self.blocks.append(block)
        self.conv_out = nn.Conv1d(hidden_dim, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        out = z
        for block in self.blocks:
            out = block(out)
        out = self.conv_out(out)
        B, C, T = out.shape
        return out.transpose(1, 2).view(B, T, -1, 3)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_joints=87,
        in_dims=3,
        hidden_dim=512,
        codebook_size=768,
        decay=0.99,
        commitment_weight=0.9,
        codebook_dim=64,
        use_cosine_sim=True,
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=2,
        orthogonal_reg_weight=1.0,
        orthogonal_reg_max_codes=256
    ):
        super().__init__()
        self.encoder = VQVAEEncoder(in_joints, in_dims, hidden_dim)
        self.quantizer = VectorQuantize(
            dim=hidden_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            codebook_dim=codebook_dim,
            use_cosine_sim=use_cosine_sim,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes
        )
        self.decoder = VQVAEDecoder(in_joints, in_dims, hidden_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_perm = z_e.permute(0, 2, 1)
        quantized, indices, vq_loss = self.quantizer(z_perm)
        q_perm = quantized.permute(0, 2, 1)
        x_recon = self.decoder(q_perm)
        return x_recon, indices, vq_loss