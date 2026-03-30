import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import CastedLinear

class LoRACastedLinear(nn.Module):
    def __init__(self, base_layer: CastedLinear, r: int = 64, alpha: int = 32, dropout: float = 0.0, layer_name: str = ""):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.layer_name = layer_name
        self.active_loradict = None
        
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
        orig_dtype = x.dtype
        x_dropped = self.lora_dropout(x)
        
        loradict = self.active_loradict
        
        if loradict is not None and f"{self.layer_name}.lora_A" in loradict:
            lora_A = loradict[f"{self.layer_name}.lora_A"].to(orig_dtype)
            lora_B = loradict[f"{self.layer_name}.lora_B"].to(orig_dtype)
            
            x_dim = x_dropped.dim()
            if x_dim == 2:
                # [Batch, in_features] -> [Batch, 1, in_features]
                x_dropped_3d = x_dropped.unsqueeze(1)
            else:
                x_dropped_3d = x_dropped
                
            # x_dropped_3d: [Batch, SeqLen, in_features]
            # lora_A.transpose(1, 2): [Batch, in_features, r]
            lora_out = torch.bmm(x_dropped_3d, lora_A.transpose(1, 2))
            
            # lora_out: [Batch, SeqLen, r]
            # lora_B.transpose(1, 2): [Batch, r, out_features]
            lora_out = torch.bmm(lora_out, lora_B.transpose(1, 2))
            
            if x_dim == 2:
                lora_out = lora_out.squeeze(1)
        else:
            lora_out = F.linear(x_dropped, self.lora_A.to(orig_dtype))
            lora_out = F.linear(lora_out, self.lora_B.to(orig_dtype))
            
        return result + lora_out * self.scaling


def inject_lora(model: nn.Module, r: int = 64, alpha: int = 32, dropout: float = 0.0, prefix: str = "") -> nn.Module:
    """
    Recursively replaces all CastedLinear modules with LoRACastedLinear inside the model.
    """
    for name, module in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, CastedLinear):
            # Wrap the existing CastedLinear with our LoRA layer
            lora_layer = LoRACastedLinear(module, r=r, alpha=alpha, dropout=dropout, layer_name=layer_name)
            setattr(model, name, lora_layer)
        else:
            # Recursively apply to children
            inject_lora(module, r=r, alpha=alpha, dropout=dropout, prefix=layer_name)
    return model
