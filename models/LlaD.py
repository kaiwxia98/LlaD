import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StandardNorm import Normalize
from layers.VTR import Projector
from layers.LAD import InterLayerAggregation
from layers.BFA import BidirectionalFeatureAlignment
from transformers import GPT2Model


class LlaD(nn.Module):
    def __init__(
        self, device="cuda:0", channel=32, num_nodes=7, seq_len=96, pred_len=96,
        dropout_n=0.1, d_llm=768, e_layer=1, d_layer=1, d_ff=32, head=8,
        model_name="gpt2", token_axis="var",
    ):
        super().__init__()
        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_llm = d_llm
        # "var": tokens = N variables (default, inverted); "time": tokens = L timesteps
        assert token_axis in ("var", "time"), f"unknown token_axis={token_axis}"
        self.token_axis = token_axis
        
        self.e_student_layer = e_layer 
        
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        if self.token_axis == "var":
            # VTR: phi_len, [B, N, L] -> [B, N, C]
            self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
            self.c_to_length = nn.Linear(self.channel, self.pred_len).to(self.device)
        else:
            # [B, L, N] -> [B, L, C]; predict via L->pred_len then C->N
            self.var_to_feature = nn.Linear(self.num_nodes, self.channel).to(self.device)
            self.time_to_pred = nn.Linear(self.seq_len, self.pred_len).to(self.device)
            self.feat_to_var = nn.Linear(self.channel, self.num_nodes).to(self.device)

        self.gpt2 = GPT2Model.from_pretrained(
            "/home/xiakaiwen/.cache/modelscope/hub/models/AI-ModelScope/gpt2",
            local_files_only=True
        )
        
        self.gpt2.to(self.device)
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # LAD: inter-layer aggregation, last GPT-2 layer attends to earlier layers
        self.layer_enhancer = InterLayerAggregation(self.d_llm).to(self.device)

        # VTR: phi_remap, TS Feature (32) -> GPT Embedding (768)
        self.ts_to_gpt_proj = Projector(self.channel, self.d_llm, dropout=dropout_n).to(self.device)
        
        # LAD: teacher feature adapter, 768 -> 768
        self.teacher_adapter = Projector(self.d_llm, self.d_llm, dropout=dropout_n, hidden_dim=self.d_llm).to(self.device)

        # Time series encoder Enc_ts
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model=self.channel, nhead=head, batch_first=True, 
                                                           norm_first=True, dropout=dropout_n, dim_feedforward=d_ff).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=e_layer).to(self.device)

        # Student Projector: 32 -> 768
        self.student_proj = Projector(self.channel, self.d_llm, dropout=dropout_n).to(self.device)
        
        # LAD: student encoder
        self.student_reasoning = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_llm, nhead=head, batch_first=True, norm_first=True, dropout=dropout_n),
            num_layers=self.e_student_layer
        ).to(self.device)

        # BFA & Decoder
        self.cross = BidirectionalFeatureAlignment(d_model_ts=self.channel, d_model_emb=self.d_llm, nhead=head, dropout=dropout_n).to(self.device)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.channel, nhead=head, batch_first=True, norm_first=True, dropout=dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=d_layer).to(self.device)
        
        self.model_name = model_name

        print("=" * 60)
        print("Current model_name:", model_name, "| token_axis:", self.token_axis)
        print("=" * 60)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _encode_features(self, input_data):
        # input_data: [B, L, N] (already RevIN-normalized)
        if self.token_axis == "var":
            # inverted: tokens = variables -> [B, N, C]
            return self.length_to_feature(input_data.permute(0, 2, 1))
        # temporal: tokens = timesteps -> [B, L, C]
        return self.var_to_feature(input_data)

    def _forecast_head(self, token_feats):
        """Map token features to [B, pred_len, N]."""
        if self.token_axis == "var":
            # token_feats: [B, N, C] -> [B, N, pred_len] -> [B, pred_len, N]
            return self.c_to_length(token_feats).permute(0, 2, 1)
        # token_feats: [B, L, C] -> [B, C, L] -> [B, C, pred_len] -> [B, pred_len, C] -> [B, pred_len, N]
        x = self.time_to_pred(token_feats.permute(0, 2, 1)).permute(0, 2, 1)
        return self.feat_to_var(x)

    def forward(self, input_data, input_data_mark, embeddings=None, mode='train'):
        # input_data: [B, L, N]
        input_data = self.normalize_layers(input_data, 'norm')
        input_features = self._encode_features(input_data)  # [B, T, C], T in {N, L}

        # === Student Branch ===
        enc_out = self.ts_encoder(input_features) # [B, T, C]
        
        # Student Embedding (Distillation Target)
        student_emb_input = self.student_proj(enc_out) 
        student_thought = self.student_reasoning(student_emb_input) # [B, T, 768]

        # === Teacher Branch ===
        teacher_thought = None
        dec_out_teacher = None

        if mode == 'train':

            # VTR: remap variable-wise features into the frozen LLM input space
            gpt_inputs = self.ts_to_gpt_proj(input_features) #  Prompt 
            # gpt_inputs = torch.zeros(input_features.shape[0], input_features.shape[1], self.d_llm).to(self.device)

            gpt_outputs = self.gpt2(inputs_embeds=gpt_inputs, output_hidden_states=True)

            # last_hidden_state as anchor, enhanced by selective attention to earlier layers
            all_hidden = torch.stack(gpt_outputs.hidden_states[1:], dim=0)  # [12, B, T, 768]
            teacher_feat = self.layer_enhancer(all_hidden)  # [B, T, 768]

            # last = all_hidden[-1]                          # [B, N, D]
            # earlier = all_hidden[:-1]                       # [n_layers-1, B, N, D]
            # ctx = earlier.mean(dim=0)                       # [B, N, D]
            # teacher_feat = last + self.layer_enhancer.out_proj(ctx)

            teacher_thought = self.teacher_adapter(teacher_feat)

            # Teacher Prediction
            # BFA returns [B, C, T]; permute back to [B, T, C]
            cross_out_t = self.cross(enc_out, teacher_thought).permute(0, 2, 1)
            dec_out_t = self.decoder(cross_out_t, cross_out_t)
            dec_out_teacher = self.normalize_layers(self._forecast_head(dec_out_t), 'denorm')

        # === Student Prediction ===
        cross_out_s = self.cross(enc_out, student_thought).permute(0, 2, 1)
        
        dec_out_s = self.decoder(cross_out_s, cross_out_s)
        dec_out_student = self.normalize_layers(self._forecast_head(dec_out_s), 'denorm')

        if mode == 'train':
            return dec_out_student, dec_out_teacher, student_thought, teacher_thought
        
        return dec_out_student
