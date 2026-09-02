import torch
import torch.nn as nn
import torch.nn.functional as F

# 可学习的位置编码+新融合方式
class BidirectionalFeatureAlignment(nn.Module):
    def __init__(self, d_model_ts, d_model_emb, nhead=4, dropout=0.1, seq_len=100):
        super().__init__()

        self.d_model_ts = d_model_ts
        self.d_model_emb = d_model_emb
        self.seq_len = seq_len

        # 投影 Embedding → TS 空间
        self.emb_proj = nn.Linear(d_model_emb, d_model_ts)

        # 改成可广播的
        self.pos_ts = nn.Parameter(torch.randn(1, 1, d_model_ts))
        self.pos_emb = nn.Parameter(torch.randn(1, 1, d_model_ts))

        # 双向 Cross Attention（DecoderLayer）
        # # 模态A→B：TS attends to Embedding（投影后）
        # self.attn_ts_to_emb = nn.TransformerDecoderLayer(
        #     d_model=d_model_ts, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True
        # )
        # # 模态B→A：Embedding attends to TS（先投影）
        # self.attn_emb_to_ts = nn.TransformerDecoderLayer(
        #     d_model=d_model_ts, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True
        # )
        self.attn_ts_to_emb = nn.MultiheadAttention(
            embed_dim=d_model_ts, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.attn_emb_to_ts = nn.MultiheadAttention(
            embed_dim=d_model_ts, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # 融合层（更深更稳定）
        self.fusion = nn.Sequential(
            nn.Linear(d_model_ts * 2, d_model_ts),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model_ts)

    def forward(self, ts_feats, emb_feats, return_weights=False):
        # 输入: [B, C, N] → [B, N, C]
        ts_feats = ts_feats + self.pos_ts  # [1,1,D] -> 广播到[B,N,D]
        emb_proj = self.emb_proj(emb_feats) + self.pos_emb

        # TS -> Emb 注意力
        ts_attended, ts_weights = self.attn_ts_to_emb(
            query=ts_feats,
            key=emb_proj,
            value=emb_proj,
            need_weights=return_weights  # 动态控制权重计算
        )
        
        # Emb -> TS 注意力
        emb_attended, emb_weights = self.attn_emb_to_ts(
            query=emb_proj,
            key=ts_feats,
            value=ts_feats,
            need_weights=return_weights
        )


        # 融合两个模态的注意力输出
        fused = torch.cat([ts_attended, emb_attended], dim=-1)      # [B, N, 2*d_model_ts]
        
        # 消融4
        # fused = torch.cat([ts_feats, emb_proj], dim=-1) 
        
        fused = self.fusion(fused)                                  # [B, N, d_model_ts]

        # 残差连接 + LayerNorm（提升训练稳定性）
        fused = self.norm(fused + ts_feats)                         # [B, N, d_model_ts]

        if return_weights:
            return fused.permute(0, 2, 1), (ts_weights, emb_weights)

        # 输出: [B, N, C] → [B, C, N]
        return fused.permute(0, 2, 1)                                # [B, d_model_ts, N]
