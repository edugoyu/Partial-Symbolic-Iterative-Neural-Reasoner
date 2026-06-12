import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Locks all random number generators and forces strict GPU determinism."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_board(n):
    """Generates a valid N-Queens board and its closest-queen color regions."""
    if n > 1 and n < 4:
        raise ValueError("A solution is not possible for N=2 or N=3 with diagonal constraints.")

    x_vals, y_vals = list(range(1, n + 1)), list(range(1, n + 1))
    valid = False
    while not valid:
        random.shuffle(y_vals)
        valid = True
        for i in range(n - 1):
            if abs(y_vals[i] - y_vals[i+1]) == 1: 
                valid = False
                break
                
    coordinates = list(zip(x_vals, y_vals))
    palette = plt.cm.tab10(np.linspace(0, 1, n))
    
    qx = np.array([x - 1 for x, y in coordinates])
    qy = np.array([y - 1 for x, y in coordinates])
    
    R, C = np.mgrid[0:n, 0:n]
    dist = (R[..., np.newaxis] - qy)**2 + (C[..., np.newaxis] - qx)**2
    grid_data = np.argmin(dist, axis=-1)
            
    return grid_data, coordinates, palette

class CustomEncoderLayer(nn.Module):
    """A custom Transformer encoder block designed to output attention weights."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, 512)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(
            src, src, src, 
            need_weights=True, 
            average_attn_weights=False
        )
        src = src + self.norm1(attn_output)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.norm2(src2)
        
        return src, attn_weights