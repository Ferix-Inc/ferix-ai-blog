import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


EPSILON = 1e-5
N_BIT = 8
QB = 2 ** (N_BIT - 1) - 1


# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor):an input tensor with shape [batch_size, seq_len, in_features]

    Returns:
        torch.Tensor: a quantized input tensor
    """
    Qb_over_gamma = QB / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=EPSILON)
    x_tilde = (x * Qb_over_gamma).round().clamp_(-QB, QB) / Qb_over_gamma
    return x_tilde


# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        w (torch.Tensor): a weight tensor with shape [in_features, out_features]

    Returns:
        torch.Tensor: a quantized weight tensor
    """
    gamma =  w.abs().mean().clamp_(min=EPSILON)
    w_tilde = (w / gamma).round().clamp_(-1, 1) * gamma
    return w_tilde


# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.layer_norm = LlamaRMSNorm(in_features)
        self.init()

    def init(self):
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): an input tensor with shape [batch_size, seq_len, in_features]

        Returns:
            torch.Tensor: an output tensor with shape [batch_size, seq_len, out_features]
        """
        x_norm = self.layer_norm(x)
        x_quant = x_norm + (activation_quant(x) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        y = F.linear(x_quant, w_quant)
        return y


if __name__ == "__main__":
    pass
