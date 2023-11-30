import torch
import torch.nn as nn
from feature_extractor import *
import numpy as np

device= 'cpu'
class SWINIR(nn.Module):
    def __init__(self, dim: tuple, RSTB_nos = 6, STL_nos = 6, window_size = (8, 8), channel_nos = 180, attn_head = 6, device = device):
    
        """Implementation of the SWINIR Model:
        
        Inputs:
        - dim: dimension of the input image (C, H, W)
        - RSTB_nos: No. of RSTB in the deep feature extractor
        - STL_nos: No. of STL in a RSTB 
        - window_size: Size of a window in the MSA
        - channel_nos: No. of feature channels
        - attn_head: No. of attention heads for the MSA"""
        
        super().__init__()
        self.extractor = feature(input_dimension = dim, output_dimension = (channel_nos,*dim[1:]), n = np.ones(RSTB_nos, dtype=int) * STL_nos, heads = np.ones((RSTB_nos, STL_nos), dtype=int) * attn_head, window_size = window_size, device=device).to(device=device)
        self.HQ_Reconstruction = nn.Sequential(
            nn.ConvTranspose2d(channel_nos, channel_nos//4, 4, 2, padding=1),
            nn.Conv2d(channel_nos//4, channel_nos//4, 3, padding="same"),
            nn.ConvTranspose2d(channel_nos//4, dim[0], 3),
            nn.Conv2d(dim[0], dim[0], 3, padding="same")
        ).to(device=device)
        
    def forward(self, img):
        return self.HQ_Reconstruction(self.extractor(img))


## Modified (Ignored for Now)
class SwinIR(nn.Module):
    def __init__(self, num_stages=5, num_blocks_per_stage=6, window_size=8, embedding_dim=96):
        super(SwinIR, self).__init__()

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = SwinTransformer(num_blocks=num_blocks_per_stage, window_size=window_size, embedding_dim=embedding_dim)
            self.stages.append(stage)

        self.reconstruction_block = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)

        x = self.reconstruction_block(x)

        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_blocks, window_size=8, embedding_dim=96):
        super(SwinTransformer, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = SwinTransformerBlock(window_size=window_size, embedding_dim=embedding_dim)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, window_size=8, embedding_dim=96):
        super(SwinTransformerBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        self.window_shift_attention = WindowShiftAttention(window_size=window_size, embedding_dim=embedding_dim)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.self_attention(x, x, x)
        x = x + x
        x = self.window_shift_attention(x, x, x)
        x = x + x
        x = self.residual_connection(x)
        x = x + x

        return x

class WindowShiftAttention(nn.Module):
    def __init__(self, window_size=8, embedding_dim=96):
        super(WindowShiftAttention, self).__init__()

        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.shift_matrix = torch.zeros((window_size, window_size), dtype=torch.int64)
        for i in range(window_size):
            for j in range(window_size):
                self.shift_matrix[i, j] = (i - window_size // 2) * window_size + j - window_size // 2

        self.shift_attention = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)

    def forward(self, x, q, k):
        b, c, h, w = x.shape

        # Shift the windows
        x_shifted = torch.roll(x, self.shift_matrix, dims=(2, 3))
        k_shifted = torch.roll(k, self.shift_matrix, dims=(2, 3))
        # Calculate the shifted attention weights
        shifted_attention = self.shift_attention(q, x_shifted, k_shifted)

        # Unshift the shifted attention weights
        attention = torch.roll(shifted_attention, -self.shift_matrix, dims=(2, 3))

        return attention
