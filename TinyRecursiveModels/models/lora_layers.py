import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import CastedLinear

class LoRACastedLinear(nn.Module):
    def __init__(self, base_layer: CastedLinear, r: int = 64, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # CastedLinear weight is (out_features, in_features)
        out_features, in_features = base_layer.weight.shape
        
        # LoRA parameters A and B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights (same as standard PEFT initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward
        result = self.base_layer(x)
        
        # LoRA forward: x @ A^T @ B^T * scaling
        # F.linear(x, weight) computes x @ weight^T
        orig_dtype = x.dtype
        lora_out = F.linear(self.lora_dropout(x), self.lora_A.to(orig_dtype))
        lora_out = F.linear(lora_out, self.lora_B.to(orig_dtype))
        
        return result + lora_out * self.scaling


def inject_lora(model: nn.Module, r: int = 64, alpha: int = 32, dropout: float = 0.0) -> nn.Module:
    """
    Recursively replaces all CastedLinear modules with LoRACastedLinear inside the model.
    """
    for name, module in model.named_children():
        if isinstance(module, CastedLinear):
            # Wrap the existing CastedLinear with our LoRA layer
            lora_layer = LoRACastedLinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer)
        else:
            # Recursively apply to children
            inject_lora(module, r=r, alpha=alpha, dropout=dropout)
    return model
