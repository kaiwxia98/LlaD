# LlaD: LLM Layer-Adaptive Distillation

Official implementation of **Towards Effective and Efficient Time Series Forecasting via LLM Layer-Adaptive Distillation**.

LlaD remaps variable-wise historical windows into a frozen LLM, adaptively aggregates multi-layer LLM features, distills them into a lightweight student encoder, and fuses the student-learned features with a time series branch for forecasting. At inference, the LLM is removed.



*Overview of LlaD (Fig. 2). VTR remaps each variable into the frozen LLM input space; LAD aggregates and distills multi-layer features into a student encoder; BFA aligns LLM-derived features with temporal features. Red paths are disabled at inference.*

## Method

LlaD has three modules, corresponding to `layers/VTR.py`, `layers/LAD.py`, and `layers/BFA.py`.

**Variable-wise Temporal Remapping (VTR).**  
Each variable’s look-back window is treated as one token. A linear map \phi_{\mathrm{len}}: \mathbb{R}^{L} \rightarrow \mathbb{R}^{C} followed by a two-layer projector \phi_{\mathrm{remap}}: \mathbb{R}^{C} \rightarrow \mathbb{R}^{D_{\mathrm{LLM}}} produces LLM input embeddings Z^{(0)}. GPT-2 is frozen and is fed `inputs_embeds` (no text prompts).

**Layer-Adaptive Distillation (LAD).**  
All GPT-2 hidden states Z^{(1)},\ldots,Z^{(M)} are kept. Inter-layer attention uses the last layer as query and earlier layers as key/value, with a zero-initialized residual projection so the aggregator starts from Z^{(M)}. The aggregated teacher feature is distilled into a shallow Transformer student with temperature-scaled KL (\tau = 3). The distillation weight decays over training.

**Bidirectional Feature Alignment (BFA).**  
Student (or teacher) LLM-derived features and the time series encoder features interact through bidirectional multi-head attention, concatenation fusion, and a residual LayerNorm, then a Transformer decoder produces the forecast.

At **inference**, GPT-2 and the teacher path are skipped. Only the student encoder, time series encoder, BFA, and decoder are used.


| Paper module                                   | Code                                                      |
| ---------------------------------------------- | --------------------------------------------------------- |
| VTR \phi_{\mathrm{len}}, \phi_{\mathrm{remap}} | `length_to_feature`, `ts_to_gpt_proj` in `models/LlaD.py` |
| LAD inter-layer aggregation                    | `layers/LAD.py` (`InterLayerAggregation`)                 |
| LAD student encoder                            | `student_proj`, `student_reasoning`                       |
| BFA                                            | `layers/BFA.py` (`BidirectionalFeatureAlignment`)         |




## Repository Layout

```
LlaD-main/
├── train.py                 # training / evaluation entry
├── models/LlaD.py           # full model
├── layers/
│   ├── VTR.py               # nonlinear projector (φ_remap / adapters)
│   ├── LAD.py               # inter-layer attention
│   ├── BFA.py               # bidirectional alignment
│   └── StandardNorm.py      # reversible instance normalization
├── data_provider/           # ETT / custom / PEMS loaders
├── utils/                   # metrics, scaler, PCA helpers
├── scripts/ETTh1.sh         # ETTh1, look-back 96 → horizon 96
├── dataset/                 # CSV files (ETTh1.csv, …)
└── figs/framework.png       # Fig. 2 from the paper
```



## Requirements

- Python 3.8+
- PyTorch with CUDA
- `transformers` (GPT-2)
- `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

Place a local GPT-2 checkpoint and set the path in `models/LlaD.py` (`GPT2Model.from_pretrained(...)`). 

## Data

Put multivariate CSV files under `dataset/`. ETTh1 is expected as `dataset/ETTh1.csv`. The loader currently reads from that directory (or the path configured in `data_provider/data_loader_emb.py`).

## Training

ETTh1 with look-back 96 and horizon 96:

```bash
cd LlaD-main
CUDA_VISIBLE_DEVICES=0 bash scripts/ETTh1.sh
```

The script writes logs to `Results/ETTh1/` and checkpoints to `logs/ETTh1/`. Equivalent command:

```bash
python -u train.py \
  --data_path ETTh1 \
  --seq_len 96 --pred_len 96 \
  --batch_size 256 --num_nodes 7 \
  --channel 64 --e_layer 1 --d_layer 2 \
  --dropout_n 0.8 --learning_rate 0.0006 \
  --epochs 100 --seed 42 --head 8 \
  --weight_decay 0.01 \
  --teacher_task_weight 0.5 --distill_weight 0.1 \
  --num_workers 4
```

Optimizer is AdamW with cosine annealing and gradient clipping (max norm 5). Checkpoints are selected by test MSE. 