import torch.nn as nn


class Projector(nn.Module):
    """Nonlinear projector used by VTR (phi_remap) and LAD (feature adapter)."""

    def __init__(self, enc_in, d_model, dropout=0.1, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2

        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)
