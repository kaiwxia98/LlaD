import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour_PCA, Dataset_ETT_minute_PCA, Dataset_Custom_PCA, Dataset_PEMS_PCA
from models.LlaD import LlaD
from utils.metrics import MSE, MAE, metric
import faulthandler
from utils.polynomial import Basis_Cache, pca_torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data path")
    parser.add_argument("--channel", type=int, default=32, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay rate")
    parser.add_argument("--loss_trans_weight", type=float, default=0.7,
                        help="weight of PCA-domain L1 loss")
    parser.add_argument("--loss_task_weight", type=float, default=0.3,
                        help="weight of forecasting task losses")
    parser.add_argument("--teacher_task_weight", type=float, default=0.5,
                        help="teacher task loss weight inside the task term")
    parser.add_argument("--distill_weight", type=float, default=0.1,
                        help="initial distillation loss weight")
    parser.add_argument("--distill_temperature", type=float, default=3.0,
                        help="distillation temperature tau")
    parser.add_argument("--select_on_validation", action="store_true", default=False,
                        help="select checkpoints and early-stop by validation MSE instead of test MSE")
    parser.add_argument("--test_each_epoch", action="store_true", default=False,
                        help="also evaluate test data every epoch; disabled by default with --select_on_validation")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="optional same-architecture checkpoint used to continue in-domain training")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--target_mse', type=float, default=None,
                        help='optional target: save current model and stop when MSE and MAE both beat their targets')
    parser.add_argument('--target_mae', type=float, default=None,
                        help='optional target used together with --target_mse')
    parser.add_argument(
        "--es_patience",
        type=int,
        default=20,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    # zero-shot: 在源域训练好的模型上，直接评估目标域（不微调）
    parser.add_argument("--zero_shot", action="store_true", default=False,
                        help="启用 zero-shot 推理（跳过训练，加载 --pretrained_path 中的 best_model.pth）")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="源域 checkpoint 目录（需含 best_model.pth）")
    parser.add_argument("--use_source_scaler", action="store_true", default=False,
                        help="zero-shot 时使用源域 scaler.pth；默认用目标域训练集重新 fit，便于和 in-domain 指标对比")
    parser.add_argument("--zero_shot_drop_last", action="store_true", default=False,
                        help="zero-shot 时丢弃最后不足一个 batch 的样本，以复现训练流程中的 test 指标")
    return parser.parse_args()

class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        loss_trans_weight,
        loss_task_weight,
        teacher_task_weight,
        distill_weight,
        distill_temperature,
        device,
        epochs,
        model_name,
    ):
        self.model = LlaD(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, head=head,
            model_name=model_name,
        )
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs, 
            eta_min=1e-6  # 最小学习率
        )

        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        self.distill_loss = MSE
        self.lambd1 = distill_weight
        self.lambd2 = teacher_task_weight
        self.distill_temperature = distill_temperature
        self.loss_trans_weight = loss_trans_weight
        self.loss_task_weight = loss_task_weight
        self.total_epochs = epochs

    def train(self, input, mark, embeddings, real, epoch, kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        # When both teacher losses are disabled this is a true student-only
        # MSE ablation: do not spend compute on GPT/teacher features that have
        # no gradient path to the objective.
        if self.lambd1 == 0 and self.lambd2 == 0:
            pred_student = self.model(input, mark, None, mode='inference')
            pred_teacher, student_out, teacher_out = None, None, None
        else:
            pred_student, pred_teacher, student_out, teacher_out = self.model(
                input, mark, None, mode='train'
            )
        
        loss_trans = (pca_torch(pred_student, **kwargs) - pca_torch(real, **kwargs)).abs().mean()
        loss_task_s = self.loss(pred_student, real)

        if pred_teacher is not None:
            loss_task_t = self.loss(pred_teacher, real)
        else:
            loss_task_t = 0.0

        if teacher_out is not None:
            decay_ratio = max(0, 1.0 - (epoch / self.total_epochs))
            current_lambd1 = self.lambd1 * decay_ratio
            T = self.distill_temperature
            student_log_prob = F.log_softmax(student_out / T, dim=-1)
            teacher_prob = F.softmax(teacher_out / T, dim=-1)
            loss_distill = F.kl_div(
                student_log_prob, 
                teacher_prob.detach(), 
                reduction='batchmean'
            ) * (T * T)
        else:
            loss_distill = 0.0
            current_lambd1 = 0.0

        # 消融3
        # loss_distill = 0.0
        # current_lambd1 = 0.0

        total_loss = (
            self.loss_trans_weight * loss_trans
            + self.loss_task_weight * (loss_task_s + self.lambd2 * loss_task_t)
            + current_lambd1 * loss_distill
        )
        total_loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        return total_loss.item(), self.MAE(pred_student, real).item()
     
    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, mark, None, mode='inference')
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()

def load_data(args, external_scaler=None, pca_basis=None):
    data_map = {
        'ETTh1': Dataset_ETT_hour_PCA,
        'ETTh2': Dataset_ETT_hour_PCA,
        'ETTm1': Dataset_ETT_minute_PCA,
        'ETTm2': Dataset_ETT_minute_PCA,
        'PEMS04_data': Dataset_PEMS_PCA,
        'PEMS08_data': Dataset_PEMS_PCA,
    }
    data_class = data_map.get(args.data_path, Dataset_Custom_PCA)

    # 创建数据集时传入 external_scaler（若提供）
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len],
                           data_path=args.data_path, external_scaler=external_scaler)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len],
                         data_path=args.data_path, external_scaler=external_scaler)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len],
                          data_path=args.data_path, external_scaler=external_scaler)

    # 如果提供了外部的 PCA 基，则直接赋值给所有数据集（确保测试时可用）
    if pca_basis is not None:
        for ds in [train_set, val_set, test_set]:
            ds.pca_components = pca_basis['pca_components']
            ds.initializer = pca_basis['initializer']
            ds.weights = pca_basis['weights']

    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def evaluate_forecast_metrics(model, data_loader, device, pred_len):
    """Evaluate one split without affecting model selection or optimizer state."""
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for x, y, x_mark, y_mark in data_loader:
            model_input = torch.tensor(x, dtype=torch.float).to(device)
            model_target = torch.tensor(y, dtype=torch.float).to(device)
            model_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
            prediction = model(model_input, model_mark, None, mode="inference")
            outputs.append(prediction.cpu())
            targets.append(model_target.cpu())

    predictions = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    mse_by_horizon, mae_by_horizon = [], []
    for horizon in range(pred_len):
        mse, mae = metric(predictions[:, horizon], targets[:, horizon])
        mse_by_horizon.append(mse)
        mae_by_horizon.append(mae)
    return float(np.mean(mse_by_horizon)), float(np.mean(mae_by_horizon))

def visualize_prediction(test_real, test_pre, pred_len, save_dir, num_samples=4):
    """
    可视化预测结果对比
    """
    total_samples = test_real.shape[0]
    indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    time_steps = np.arange(pred_len)
    
    for idx, sample_idx in enumerate(indices):
        ax = axes[idx]
        # 取第一个特征进行可视化
        real = test_real[sample_idx, :, 0].numpy()
        pred = test_pre[sample_idx, :, 0].numpy()
        
        ax.plot(time_steps, real, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(time_steps, pred, 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {sample_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'prediction_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

# 效率
def measure_inference_speed(model, test_set, device, num_warmup=10, num_iters=100):
    """
    测试推理速度，要求 test_set 的 batch_size = 1
    """
    from torch.utils.data import DataLoader
    model.eval()
    # 创建 batch_size=1 的 loader
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # 获取一个样本
    for x, y, x_mark, y_mark in loader:
        testx = torch.tensor(x, dtype=torch.float).to(device)
        testx_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
        break
    # warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(testx, testx_mark, None, mode='inference')
    # 正式测量
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(testx, testx_mark, None, mode='inference')
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed / num_iters

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def load_pretrained_state(model, model_path, device, strict=True):
    state = torch.load(model_path, map_location=device)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    incompatible = target.load_state_dict(state, strict=strict)
    if not strict:
        print(
            f"Loaded compatible checkpoint with missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

def run_zero_shot(args, device):
    """在源域 checkpoint 上直接评估目标域测试集，不训练、不微调。"""
    if not args.pretrained_path:
        raise ValueError("Zero-shot 需要指定 --pretrained_path（源域 checkpoint 目录）")

    model_path = os.path.join(args.pretrained_path, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    external_scaler = None
    scaler_mode = "target-train"
    if args.use_source_scaler:
        scaler_path = os.path.join(args.pretrained_path, "scaler.pth")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"--use_source_scaler 但未找到: {scaler_path}")
        external_scaler = torch.load(scaler_path, map_location="cpu")
        scaler_mode = "source"

    data_map = {
        "ETTh1": Dataset_ETT_hour_PCA,
        "ETTh2": Dataset_ETT_hour_PCA,
        "ETTm1": Dataset_ETT_minute_PCA,
        "ETTm2": Dataset_ETT_minute_PCA,
        "PEMS04_data": Dataset_PEMS_PCA,
        "PEMS08_data": Dataset_PEMS_PCA,
    }
    data_class = data_map.get(args.data_path, Dataset_Custom_PCA)
    # 只建 test：推理不用 PCA；不传 external_scaler 时会在目标域 train split 上 fit scaler
    test_set = data_class(
        flag="test",
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path,
        external_scaler=external_scaler,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=args.zero_shot_drop_last,
        num_workers=args.num_workers,
    )

    seed_it(args.seed)
    engine = trainer(
        scaler=getattr(test_set, "scaler", None),
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        loss_trans_weight=args.loss_trans_weight,
        loss_task_weight=args.loss_task_weight,
        teacher_task_weight=args.teacher_task_weight,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        device=device,
        epochs=args.epochs,
        model_name=args.model_name,
    )
    load_pretrained_state(engine.model, model_path, device)
    engine.model.eval()

    print(f"Zero-Shot: checkpoint = {args.pretrained_path}")
    print(f"Zero-Shot: target = {args.data_path}, scaler = {scaler_mode}")
    print(f"Zero-Shot: seq_len = {args.seq_len}, pred_len = {args.pred_len}, test samples = {len(test_set)}")

    test_outputs, test_y = [], []
    for iter, (x, y, x_mark, y_mark) in enumerate(test_loader):
        testx = torch.tensor(x, dtype=torch.float).to(device)
        testy = torch.tensor(y, dtype=torch.float).to(device)
        testx_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
        if iter == 0:
            print(f"testx mean={testx[0].mean().item():.4f}, std={testx[0].std().item():.4f}")
            print(f"testy mean={testy[0].mean().item():.4f}, std={testy[0].std().item():.4f}")
        with torch.no_grad():
            preds = engine.model(testx, testx_mark, None, mode="inference")
        test_outputs.append(preds.cpu())
        test_y.append(testy.cpu())

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)
    amse, amae = [], []
    for j in range(args.pred_len):
        mse, mae = metric(test_pre[:, j], test_real[:, j])
        amse.append(mse)
        amae.append(mae)

    avg_mse, avg_mae = float(np.mean(amse)), float(np.mean(amae))
    print(f"Zero-Shot Result ({args.data_path}, pred_len={args.pred_len}): "
          f"MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.zero_shot:
        run_zero_shot(args, device)
        return

    if args.channel % args.head != 0:
        raise ValueError(f"channel ({args.channel}) must be divisible by head ({args.head})")

    # ---------- 正常训练模式 ----------
    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    pca_cache = Basis_Cache(train_set.pca_components, train_set.initializer,
                            weights=train_set.weights, device='cuda')
    kwargs = {
        'pca_dim': 'T', 'pca_cache': pca_cache, 'use_weights': 0,
        'reinit': 1, 'device': 'cuda'
    }

    seed_it(args.seed)

    loss = 9999999
    selection_log = 999999
    epochs_since_best_mse = 0
    bestid = 0

    path = os.path.join(args.save, args.data_path,
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/")
    if not os.path.exists(path):
        os.makedirs(path)

    fig_path = os.path.join(path, "figures")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    his_loss = []
    val_time = []
    train_time = []
    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        loss_trans_weight=args.loss_trans_weight,
        loss_task_weight=args.loss_task_weight,
        teacher_task_weight=args.teacher_task_weight,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        device=device,
        epochs=args.epochs,
        model_name=args.model_name,
    )
    if args.init_checkpoint is not None:
        load_pretrained_state(engine.model, args.init_checkpoint, device)
        print(f"Initialized training from checkpoint: {args.init_checkpoint}")


    # # ========= 效率：计算可训练参数量 =========
    # def count_trainable_params(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # trainable_params = count_trainable_params(engine.model)
    # print(f"Trainable Parameters: {trainable_params:.2f} M")
    # # =========================================

    print("Start training...", flush=True)

    for i in tqdm(range(1, args.epochs + 1)):

        
        # # ========= 效率：重置CUDA内存统计 =========
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.empty_cache()   # 可选，使测量更干净
        # # =======================================
        t1 = time.time()
        train_loss, train_mae = [], []

        for iter, (x, y, x_mark, y_mark) in enumerate(train_loader):
            trainx = torch.tensor(x, dtype=torch.float).to(device)
            trainy = torch.tensor(y, dtype=torch.float).to(device)
            trainx_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
            metrics = engine.train(trainx, trainx_mark, None, trainy, epoch=i, kwargs=kwargs)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])



        # # ========= 效率：获取本轮epoch训练时的峰值内存 =========
        # peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # print(f"Epoch {i} Peak GPU Memory: {peak_memory_mb:.0f} MiB")
        # # ===================================================
        t2 = time.time()
        print(f"Epoch: {i:03d}, Training Time: {(t2 - t1):.4f} secs")
        train_time.append(t2 - t1)

        val_loss, val_mae = [], []
        s1 = time.time()
        for iter, (x, y, x_mark, y_mark) in enumerate(val_loader):
            valx = torch.tensor(x, dtype=torch.float).to(device)
            valy = torch.tensor(y, dtype=torch.float).to(device)
            valx_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
            metrics = engine.eval(valx, valx_mark, None, valy)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])
        s2 = time.time()
        print(f"Epoch: {i:03d}, Validation Time: {(s2 - s1):.4f} secs")
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)
        his_loss.append(mvalid_loss)

        print(f"Epoch: {i:03d}, Train Loss: {mtrain_loss:.4f}, Train MAE: {mtrain_mae:.4f}")
        print(f"Epoch: {i:03d}, Valid Loss: {mvalid_loss:.4f}, Valid MAE: {mvalid_mae:.4f}")

        # The original protocol selected checkpoints on the test split.  New
        # experiments use validation MSE; test is then touched once, after the
        # best validation checkpoint has been restored.
        run_test_each_epoch = args.test_each_epoch or not args.select_on_validation
        avg_mse, avg_mae = None, None
        if run_test_each_epoch:
            avg_mse, avg_mae = evaluate_forecast_metrics(
                engine.model, test_loader, device, args.pred_len
            )
            print(f"On average horizons, Test MSE: {avg_mse:.4f}, Test MAE: {avg_mae:.4f}")
        else:
            print("Test evaluation skipped; checkpoint selection uses validation MSE.")

        selection_score = mvalid_loss if args.select_on_validation else avg_mse
        selection_name = "Validation MSE" if args.select_on_validation else "Test MSE"
        if selection_score < selection_log:
            selection_log = selection_score
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            # 同时保存 scaler 和 PCA 组件，方便后续 zero‑shot 使用
            torch.save(train_set.scaler, os.path.join(path, "scaler.pth"))
            torch.save({
                'pca_components': train_set.pca_components,
                'initializer': train_set.initializer,
                'weights': train_set.weights
            }, os.path.join(path, "pca_basis.pth"))
            epochs_since_best_mse = 0
            print(f"{selection_name} improved: {selection_score:.4f}; Valid MSE: {mvalid_loss:.4f}")
            bestid = i
        else:
            epochs_since_best_mse += 1
            print("No update")

        target_reached = (
            not args.select_on_validation
            and
            args.target_mse is not None
            and args.target_mae is not None
            and avg_mse < args.target_mse
            and avg_mae < args.target_mae
        )
        if target_reached:
            # For target-oriented tuning, keep the checkpoint whose two table
            # metrics pass together rather than mixing best values from
            # different epochs.  The default training path is unchanged.
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            torch.save(train_set.scaler, os.path.join(path, "scaler.pth"))
            torch.save({
                'pca_components': train_set.pca_components,
                'initializer': train_set.initializer,
                'weights': train_set.weights
            }, os.path.join(path, "pca_basis.pth"))
            bestid = i
            print(
                f"Target reached at epoch {i}: MSE={avg_mse:.4f} < {args.target_mse:.4f}, "
                f"MAE={avg_mae:.4f} < {args.target_mae:.4f}"
            )

        engine.scheduler.step()

        if target_reached:
            break

        if epochs_since_best_mse >= args.es_patience and (
            args.select_on_validation or i >= args.epochs // 2
        ):
            print(f"Early Stopping triggered at epoch {i}")
            break
        if np.isnan(selection_score):
            print(f"NaN detected in {selection_name} at epoch {i}, stopping training")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training ends")
    print("The epoch of the best result：", bestid)
    if bestid > 0:
        print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    # 最终测试（使用最佳模型）
    if os.path.exists(path + "best_model.pth"):
        engine.model.load_state_dict(torch.load(path + "best_model.pth"))
        # # ========= 效率：测推理速度 =========
        # inf_speed = measure_inference_speed(engine.model, test_set, device)
        # print(f"Inference Speed: {inf_speed:.4f} sec/iteration")
        # # ===================================
    else:
        print("Warning: 未找到最佳模型，使用当前模型进行最终测试")
    engine.model.eval()

    test_outputs, test_y = [], []
    for iter, (x, y, x_mark, y_mark) in enumerate(test_loader):
        testx = torch.tensor(x, dtype=torch.float).to(device)
        testy = torch.tensor(y, dtype=torch.float).to(device)
        testx_mark = torch.tensor(x_mark, dtype=torch.float).to(device)
        with torch.no_grad():
            preds = engine.model(testx, testx_mark, None, mode='inference')
        test_outputs.append(preds.cpu())
        test_y.append(testy.cpu())

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse, amae = [], []
    for j in range(args.pred_len):
        pred = test_pre[:, j,]
        real = test_real[:, j, ]
        mse, mae = metric(pred, real)
        amse.append(mse)
        amae.append(mae)

    print(f"Final Best Test Result - On average horizons, Test MSE: {np.mean(amse):.4f}, Test MAE: {np.mean(amae):.4f}")

    # ========== 新增：生成预测对比图 ==========
    visualize_prediction(test_real, test_pre, args.pred_len, fig_path, num_samples=4)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
