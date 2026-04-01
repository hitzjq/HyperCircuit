import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import CastedLinear

class LoRACastedLinear(nn.Module):
    def __init__(self, base_layer: CastedLinear, r: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        
        # We don't initialize nn.Parameter here!
        # Instead, we will store them during the dynamic pass
        self.dynamic_lora_A = None
        self.dynamic_lora_B = None
        self.dynamic_scale = None
        
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def set_dynamic_lora(self, lora_A: torch.Tensor, lora_B: torch.Tensor, scale: float):
        """
        lora_A: [B, rank, in_features]
        lora_B: [B, out_features, rank]
        """
        self.dynamic_lora_A = lora_A
        self.dynamic_lora_B = lora_B
        self.dynamic_scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward -> x: [B, S, in_features]
        result = self.base_layer(x)
        
        # If dynamic LoRA parameters are loaded
        if self.dynamic_lora_A is not None and self.dynamic_lora_B is not None:
            orig_dtype = x.dtype
            x_dropped = self.lora_dropout(x).to(orig_dtype)
            
            lora_A = self.dynamic_lora_A.to(orig_dtype) # [B, rank, in_features]
            lora_B = self.dynamic_lora_B.to(orig_dtype) # [B, out_features, rank]
            
            initial_shape = x_dropped.shape
            B = initial_shape[0]
            in_features = initial_shape[-1]
            out_features = lora_B.shape[1]
            
            # Flatten to 3D for bmm: [B, ..., in_features] -> [B, N, in_features]
            x_flat = x_dropped.view(B, -1, in_features)
            
            # Batch matrix multiplication
            # x_flat: [B, N, in_features]
            # lora_A^T:  [B, in_features, rank]
            # h:         [B, N, rank]
            h = torch.bmm(x_flat, lora_A.transpose(1, 2))
            
            # h:         [B, N, rank]
            # lora_B^T:  [B, rank, out_features]
            # lora_out:  [B, N, out_features]
            lora_out = torch.bmm(h, lora_B.transpose(1, 2))
            
            # Reshape back: [B, N, out_features] -> [B, ..., out_features]
            final_shape = list(initial_shape)
            final_shape[-1] = out_features
            lora_out = lora_out.view(*final_shape)
            
            result = result + lora_out * self.dynamic_scale
            
        return result


def inject_lora(model: nn.Module, r: int = 16, dropout: float = 0.0) -> tuple[nn.Module, dict]:
    """
    Recursively replaces all CastedLinear modules with LoRACastedLinear inside the model.
    Returns:
        model: The injected model.
        lora_modules: A dict mapping full module names to their LoRACastedLinear instance,
                      allowing O(1) dynamic dict loading.
    """
    lora_modules = {}
    
    # We iterate over a list so we can mutate the children without error
    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, CastedLinear):
                # Construct the full dot-path name
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Only inject once
                if not isinstance(child_module, LoRACastedLinear):
                    lora_layer = LoRACastedLinear(child_module, r=r, dropout=dropout)
                    setattr(module, child_name, lora_layer)
                    lora_modules[full_name] = lora_layer

    return model, lora_modules
