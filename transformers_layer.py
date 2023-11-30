import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = 'cpu'
class MultiHeadAttention(nn.Module):
    """
    Usage:
      attn = MultiHeadAttention(dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, no. of windows, a specific window, size of a window)
      self_attn_output = attn(query=data, key=data, value=data)
    """

    def __init__(self, dim, num_heads, window_size, dropout=0.0, device=device):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - dim: Dimension of the window in input
         - num_heads: Number of attention heads
         - window_size: Dimension of a window (MxM)
         - dropout: Dropout probability
        """
        super().__init__()
        assert dim[0] % num_heads == 0

        self.n_head = num_heads
        self.head_dim = (dim[0])// self.n_head
        self.window = window_size[0] * window_size[1]
        
        
        self.attn_drop = nn.Dropout(dropout)
        
        self.key = nn.Linear(dim[0], dim[0]).to(device=device)
        self.query = nn.Linear(dim[0], dim[0]).to(device=device)
        self.value = nn.Linear(dim[0], dim[0]).to(device=device)
        self.B = torch.zeros(self.window, self.window).requires_grad_().to(device=device)

    def forward(self, x):
       
        dim = x.shape[1:]
        
        assert (dim[1] * dim[2]) % (self.window) == 0
        self.local_window = (dim[1] * dim[2]) // (self.window)
       
        shape = x.shape
        x = x.moveaxis((1,2,3),(3,1,2))
        shape_ = x.shape
        x = x.reshape(shape[0], self.local_window, self.window, shape[1])

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        Q = query.view(query.shape[0],  query.shape[1], query.shape[2], self.n_head, self.head_dim).moveaxis((2,3),(3,2))
        K = key.view(query.shape[0],  query.shape[1], query.shape[2], self.n_head, self.head_dim).moveaxis((2,3),(3,2))
        V = value.view(query.shape[0],  query.shape[1], query.shape[2], self.n_head, self.head_dim).moveaxis((2,3),(3,2))
        
        scores = torch.matmul(Q, K.transpose(3,4))/(self.head_dim**(1/2)) + self.B
        
        probs = F.softmax(scores,dim=-1)
        output = (torch.matmul(self.attn_drop(probs), V).moveaxis((2,3),(3,2)).reshape(shape_).moveaxis((1,2,3),(2,3,1)))

        return output
      

