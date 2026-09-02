import torch
import torch.nn as nn
import torch.nn.functional as F


class InterLayerAggregation(nn.Module):
    """Inter-layer attention in the Layer-Adaptive Distillation (LAD) module.

    Uses the last LLM layer as query and preceding layers as key/value.
    The output projection is zero-initialized so the residual starts as Z^(M).
    """

    def __init__(self, d_llm):
        super().__init__()
        d_attn = d_llm // 4
        self.query = nn.Linear(d_llm, d_attn)
        self.key = nn.Linear(d_llm, d_attn)
        self.value = nn.Linear(d_llm, d_llm)
        self.scale = d_attn ** -0.5

        # Zero-init: out_proj(x)=0 at start, so output = last + 0 = last
        self.out_proj = nn.Linear(d_llm, d_llm)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, all_hidden):
        """
        Args:
            all_hidden: [n_layers, B, N, d_llm] stacked GPT-2 hidden states
        Returns:
            [B, N, d_llm] = last_hidden_state + learned enhancement from earlier layers
        """
        last = all_hidden[-1]                          # [B, N, D]
        earlier = all_hidden[:-1].permute(1, 2, 0, 3)  # [B, N, n_earlier, D]

        Q = self.query(last).unsqueeze(2)               # [B, N, 1, d_attn]
        K = self.key(earlier)                            # [B, N, n_earlier, d_attn]
        V = self.value(earlier)                          # [B, N, n_earlier, D]

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, N, 1, n_earlier]
        attn = F.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, V).squeeze(2)          # [B, N, D]

        return last + self.out_proj(ctx)
