"""端到端入口: 数据 → 训练 → 评估 → 可视化。"""

import os
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from gnn.config import FC1_CONFIG, FC2_CONFIG, TRAIN_CONFIG, TrainConfig
from gnn.data import load_dataset
from gnn.model import GNN
from gnn.model_extensions import DeepGNN
from gnn.train import train


def _ensure_backend() -> None:
    """若无 GUI 显示能力则退回 Agg, 保证 plt.show() 在无头环境不崩。"""
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    elif os.name == "nt":
        # Windows: 若已显式设为 Agg 则保持, 否则尝试交互式。
        try:
            if matplotlib.get_backend() == "Agg":
                return
        except Exception:
            pass


def evaluate(model: GNN, data: dict) -> dict:
    """评估训练后的模型。

    Args:
        model: 训练后的 GNN。
        data: load_dataset 返回的字典。

    Returns:
        dict with keys:
            y_pred (7,), y_true (7,),
            MAPE (float), MSE (float), RMSE (float),
            a (float), bs (5,),
    """
    pred = model.predict(data)
    y_pred = pred["y_pred"]        # (7,) 全序列预测
    y_true = pred["y_true"]        # (7,) 全序列真实值

    # 只在验证集上计算指标 (IAGO 索引 = 原始索引 - 1)
    val_iago_idx = data["val_idx"] - 1   # [5, 6] for 8-point time-ordered
    y_pred_val = y_pred[val_iago_idx]
    y_true_val = y_true[val_iago_idx]

    # MAPE
    mape = np.mean(np.abs((y_true_val - y_pred_val) / y_true_val)) * 100.0
    # MSE
    mse = np.mean((y_true_val - y_pred_val) ** 2)
    # RMSE
    rmse = np.sqrt(mse)

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "MAPE": float(mape),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "a": pred["a"],
        "bs": pred["bs"],
        "val_idx": data["val_idx"],  # 验证集原始索引
    }


def plot_degradation(
    results: dict,
    data: dict,
    save_path: str | None = None,
) -> None:
    """绘制电压退化曲线: 预测 vs 真实。

    Args:
        results: evaluate 返回的字典。
        data: load_dataset 返回的字典。
        save_path: 如果给定, 保存图片到此路径。
    """
    _ensure_backend()

    char_times = np.array(data["char_times"])
    y_pred = results["y_pred"]    # (7,) for k=1..7
    y_true = results["y_true"]    # (7,)

    fig, ax = plt.subplots(figsize=(8, 5))

    # 预测和真实 (k=1..7)
    ax.plot(char_times[1:], y_true, "ko-", label="Actual", linewidth=2, markersize=6)
    ax.plot(char_times[1:], y_pred, "r^--", label="Predicted", linewidth=2, markersize=6)

    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Stack Voltage (V)", fontsize=12)
    ax.set_title(f"PEMFC Degradation Prediction — {data['name']}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 指标标注
    mape = results["MAPE"]
    rmse = results["RMSE"]
    ax.text(
        0.98, 0.95,
        f"MAPE = {mape:.4f}%\nRMSE = {rmse:.6f} V",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  plot saved to {save_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_loss(
    losses: list[float],
    config: TrainConfig,
    save_path: str | None = None,
) -> None:
    """绘制训练 loss 曲线。

    Args:
        losses: train 返回的 loss 列表。
        config: 训练配置。
        save_path: 如果给定, 保存图片到此路径。
    """
    _ensure_backend()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(losses)), losses, linewidth=1)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(f"Training Loss — {config.epochs} epochs, lr={config.lr}", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def print_metrics(results: dict) -> None:
    """打印评估指标和灰系数。"""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  MAPE : {results['MAPE']:.4f} % (validation only)")
    print(f"  MSE  : {results['MSE']:.8f} V^2 (validation only)")
    print(f"  RMSE : {results['RMSE']:.8f} V  (validation only)")
    print(f"{'='*60}")
    print(f"  Grey Coefficients")
    print(f"{'='*60}")
    print(f"  a  (development)   : {results['a']:+.6f}")
    bs = results["bs"]
    names = ["b1 (voltage)", "b2 (humidity)", "b3 (H2 press)", "b4 (temp)", "b5 (current)"]
    for name, val in zip(names, bs):
        print(f"  {name:18s}: {val:+.6f}")
    print(f"{'='*60}")
    print(f"  Predictions vs Actual")
    print(f"{'='*60}")
    val_idx = results.get("val_idx", np.array([6, 7]))
    for i in range(len(results["y_pred"])):
        tag = " [VAL]" if (i + 1) in val_idx else ""
        print(
            f"  point {i+1}{tag}:  pred={results['y_pred'][i]:.6f} V  "
            f"actual={results['y_true'][i]:.6f} V  "
            f"error={abs(results['y_pred'][i]-results['y_true'][i]):.6f} V"
        )


def main(
    dataset_name: str = "FC1",
    seed: int = 42,
    deep_lb: bool = False,
    deep_la: bool = False,
    residual: bool = False,
) -> dict:
    """端到端入口。

    Args:
        dataset_name: "FC1" 或 "FC2"。
        seed:         随机种子。
        deep_lb:      启用深层特征编码器 (MLP 5→16→8→1)。
        deep_la:      启用时变衰减率 a(k)。
        residual:     启用显式灰趋势 + 深度残差。

    Returns:
        results 字典。
    """
    torch.manual_seed(seed)

    cfg = FC1_CONFIG if dataset_name.upper() == "FC1" else FC2_CONFIG
    train_cfg = TRAIN_CONFIG

    use_deep = deep_lb or deep_la or residual
    flags = "+".join(
        n for n, v in [("deep_lb", deep_lb), ("deep_la", deep_la), ("residual", residual)] if v
    ) or "none"

    print(f"\n{'='*60}")
    print(f"  GNN — {'DeepGNN' if use_deep else 'GNN'}  [{flags}]")
    print(f"  Dataset: {cfg.name}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    # 1. 数据
    print("\n[1/4] Loading data...")
    data = load_dataset(cfg, seed=seed)
    print(f"  Train points: {len(data['y_train_ago'])}, Val points: {len(data['y_val_ago'])}")

    # 2. 模型
    print("\n[2/4] Building model...")
    if use_deep:
        model = DeepGNN(
            input_dim=5,
            hidden_dim=train_cfg.hidden_nodes,
            deep_lb=deep_lb,
            deep_la=deep_la,
            residual=residual,
        )
    else:
        model = GNN(
            input_dim=5,
            hidden_dim=train_cfg.hidden_nodes,
        )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params}")

    # 3. 训练
    print(f"\n[3/4] Training ({train_cfg.epochs} epochs, Adam, lr={train_cfg.lr})...")
    model, losses = train(model, data, train_cfg)

    # 4. 评估
    print("\n[4/4] Evaluating...")
    results = evaluate(model, data)
    print_metrics(results)

    # 5. 绘图
    plot_degradation(results, data)
    plot_loss(losses, train_cfg)

    return results


if __name__ == "__main__":
    main("FC1", seed=42)
