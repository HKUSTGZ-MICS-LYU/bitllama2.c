import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Reference: https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
'''

def map_w(w, QuantType: str = "1.5b"):
    if QuantType == "1b":
        if w == -1:
            return "1"
        elif w == 1:
            return "0"
    else:
        if w == -1:
            return "11"
        elif w == 0:
            return "00"
        elif w == 1:
            return "01"
        else:
            if QuantType == "2b":
                return "10"
            else:
                raise ValueError("Invalid value")

# def activation_quant(x: torch.Tensor):

#     scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
#     y = (x * scale).round().clamp_(-128, 127) / scale
#     return y

def activation_nquant(x: torch.Tensor, qbit = 8):
    scale = (2**(qbit-1) - 1) / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-(2**(qbit-1)), 2**(qbit-1) - 1) / scale
    return y

def weight_quant(w: torch.Tensor):
    # 1.5-bit quantization
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u, 1/scale

def weight_quantnb(w: torch.Tensor, qbit = 2):
    assert qbit > 1, "qbit should be larger than 1"
    # scale = (2**(qbit-1) - 1) / w.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5) # PreOutput
    scale = (2**(qbit-1) - 1) / w.abs().mean().clamp_(min=1e-5) # PreTensor
    u = (w * scale).round().clamp_(-(2**(qbit-1)), 2**(qbit-1) - 1)
    return u, 1/scale

def weight_quant1b(w: torch.Tensor):
    # 1-bit quantization
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = w.sign()
    return u, 1/scale

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale

class BitLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 device=None, dtype=None,
                 qtype = "1.5b",
                 act_q = 8) -> None:
        
        self.qtype = qtype
        self.act_q = act_q
        super().__init__(in_features, out_features, bias, device, dtype)

    def weight_quant(self, w: torch.Tensor):
        if self.qtype == "1.5b":
            u,s = weight_quant(w)
        elif self.qtype == "2b":
            u,s = weight_quantnb(w, 2)
        elif self.qtype == "1b":
            u,s = weight_quant1b(w)
        else:
            raise ValueError("Invalid quantization type")
        return u*s

    def export(self):
        if self.qtype == "1.5b":
            u,s = weight_quant(self.weight)
        elif self.qtype == "2b":
            u,s = weight_quantnb(self.weight, 2)
        elif self.qtype == "1b":
            u,s = weight_quant1b(self.weight)
        else:
            raise ValueError("Invalid quantization type")
        
        buffer = bytearray()
        qweight = []
        u = u.cpu().view(-1).to(torch.int8).numpy()
        for w in u:
            qweight.append(map_w(w, self.qtype))
        wb = "".join(qweight)
        i = 0
        while i < len(wb):
            buffer.append( int(wb[i:i+8], 2) )
            i += 8

        return s, buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-Through-Estimator
        w = self.weight
        x_norm = SimpleRMSNorm(self.in_features)(x)
        x_quant = x_norm + (activation_nquant(x_norm, self.act_q) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
