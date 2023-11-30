import torch
import torch.nn as nn
import RSTB

device = 'cpu'

class shallow_feature(nn.Module):
    """Class Containing the shallow feature extractor"""
    def __init__(self, input_dimension, output_dimension, device=device):
        """
        Inputs:
        - input_dimension: Tuple containing the dimensions of the input features of the image
        - output_dimension:Tuple containing the dimensions of the output features of the image"""
        super().__init__()
        self.Convolution = nn.Conv2d(in_channels = input_dimension[0], out_channels = output_dimension[0], kernel_size = 3, padding = "same").to(device=device)
        
    def forward(self, image):
        """Forward function of the shallow feature extractor"""
        return self.Convolution(image)

class deep_features(nn.Module):
    "Class containing the deep feature extractor"
    def __init__(self, n, heads, dim, window_size, device = device):
        """Inputs:
        - n : An array having the no. of elements in each RSTB
        - heads: An array of having an array as each element for heads
        - dim: dimension of the input features
        - window_size: tuple of the window size of the transformer"""
        super().__init__()
        self.layers = {}
        self.n  = len(n)
        for i in range(1, len(n)+1):
            self.layers[i] = RSTB.RSTB(dim, n[i-1], heads[i-1], window_size, device=device).to(device=device)
        self.Convolution = nn.Conv2d(dim[0], dim[0], 3, padding = 'same').to(device=device)
        
    def forward(self, features):
        x = features
        for i in range(1, self.n + 1):
            x = self.layers[i](x)
        return self.Convolution(x) + features            
        
class feature(nn.Module):
    "Feature extractor model of the Transformer"
    def __init__(self, input_dimension, output_dimension, n, heads, window_size, device=device):
        """Check the initialisation of the deep_feature and shallow_feature extractor functions"""
        super().__init__()
        self.shallow = shallow_feature(input_dimension, output_dimension, device=device).to(device=device)
        self.deep = deep_features(n, heads, output_dimension, window_size, device=device).to(device=device)
        
    def forward(self, img):
        return self.deep(self.shallow(img))
    
