"""数据加载与预处理模块。

流水线：
  原始 xlsx/csv → 拼接 → 提取特征化时刻 → 归一化 → AGO → 训练/验证划分
"""

import numpy as np
import pandas as pd
from pathlib import Path
from gnn.config import DatasetConfig, DATA_ROOT


def _read_single_file(filepath: Path) -> pd.DataFrame:
    """根据扩展名读 xlsx 或欧式 CSV（分号分隔，逗号小数点）。"""
    if filepath.suffix == ".csv":
        df = pd.read_csv(
            filepath,
            sep=";",
            decimal=",",
            encoding="utf-8",
        )
    else:
        df = pd.read_excel(filepath)
    return df


def load_raw_data(cfg: DatasetConfig) -> pd.DataFrame:
    """加载并纵向拼接一个数据集的所有 Ageing 文件。"""
    source_dir = DATA_ROOT / cfg.source_dir
    frames = []
    for fname in cfg.file_parts:
        filepath = source_dir / fname
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        df = _read_single_file(filepath)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def extract_char_points(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    """提取特征化时刻的行。

    对每个 char_time，在 Time (h) 列中找最接近的行。返回包含特征列和目标列的 DataFrame。
    """
    time_col = "Time (h)"
    if time_col not in df.columns:
        raise KeyError(f"数据中缺少时间列: {time_col}")

    selected = []
    for t in cfg.char_times:
        idx = (df[time_col] - t).abs().idxmin()
        row = df.loc[idx]
        selected.append(row)

    result = pd.DataFrame(selected, columns=df.columns)
    return result.reset_index(drop=True)


def normalize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max 归一化到 [0, 1]。

    Returns:
        normalized: 归一化后的数据
        dmin: 原始最小值
        dmax: 原始最大值
    """
    dmin = data.min(axis=0, keepdims=True)
    dmax = data.max(axis=0, keepdims=True)
    denom = dmax - dmin
    denom[denom == 0] = 1.0  # 防止除零
    normalized = (data - dmin) / denom
    return normalized, dmin, dmax


def inverse_normalize(
    normalized: np.ndarray,
    dmin: np.ndarray,
    dmax: np.ndarray,
) -> np.ndarray:
    """逆归一化：将 [0,1] 数据还原到原始尺度。"""
    return normalized * (dmax - dmin) + dmin


def ago_sequence(y: np.ndarray) -> np.ndarray:
    """一阶累加生成（AGO），论文公式 2。

    y: shape (n,) 或 (n, 1) — 归一化后的目标序列
    Returns: shape same as input
    """
    y = np.asarray(y).ravel()
    return np.cumsum(y)


def iago_sequence(y_ago: np.ndarray) -> np.ndarray:
    """逆累加生成（IAGO），论文公式 8。

    y_ago: shape (n,) — AGO 预测值
    Returns: shape (n-1,)
    """
    y_ago = np.asarray(y_ago).ravel()
    return np.diff(y_ago)


def build_samples(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """构建灰系统回归样本。

    每个特征化时刻 t 作为一个样本：
      X[t] = 时刻 t 的 5 个特征（含当前电压作为"滞后电压"）
      y[t] = 时刻 t 的目标电压

    AGO 在后续步骤中对 y 做累加，将时序依赖编码进 x^(1)。
    返回 8 组样本：X shape (8, 5), y_raw shape (8,)
    """
    features = df[cfg.feature_cols].values.astype(np.float64)
    target = df[cfg.target_col].values.astype(np.float64)

    n_expected = len(cfg.char_times)
    if len(features) != n_expected:
        raise ValueError(
            f"特征化点数量不匹配: 期望 {n_expected}，实际 {len(features)}"
        )

    return features, target


def load_dataset(
    cfg: DatasetConfig,
    seed: int = 42,
) -> dict:
    """完整数据流水线。

    Returns:
        dict with keys:
          X_train, X_val: 训练/验证特征  (n_train, 5), (n_val, 5)
          y_train_ago, y_val_ago: 训练/验证 AGO 目标
          y_train_raw, y_val_raw: 训练/验证原始目标
          X_all, y_all_ago, y_all_raw: 全部 8 个点的数据
          sort_indices: 将 shuffle 后的索引恢复到原始时间顺序的映射
          target_dmin, target_dmax: 目标列归一化参数
          char_times: 特征化时刻列表
          name: 数据集名称
    """
    # 1. 加载原始数据
    df = load_raw_data(cfg)

    # 2. 提取特征化点
    char_df = extract_char_points(df, cfg)

    # 3. 构建样本
    X, y_raw = build_samples(char_df, cfg)  # X: (8, 5), y_raw: (8,)

    # 4. 归一化特征
    X_norm, X_dmin, X_dmax = normalize(X)

    # 5. 归一化目标
    y_raw_2d = y_raw.reshape(-1, 1)
    y_norm, y_dmin, y_dmax = normalize(y_raw_2d)
    y_norm = y_norm.ravel()

    # 6. AGO 累加目标
    y_ago = ago_sequence(y_norm)  # shape: (8,)

    # 7. 划分训练/验证（6 train + 2 val）
    np.random.seed(seed)
    indices = np.arange(8)
    np.random.shuffle(indices)
    train_idx = np.sort(indices[:6])
    val_idx = np.sort(indices[6:])

    # 用于恢复原始时间顺序
    sort_indices = np.argsort(indices)

    return {
        "X_train": X_norm[train_idx],
        "X_val": X_norm[val_idx],
        "y_train_ago": y_ago[train_idx],
        "y_val_ago": y_ago[val_idx],
        "y_train_raw": y_raw[train_idx],
        "y_val_raw": y_raw[val_idx],
        "X_all": X_norm,
        "y_all_ago": y_ago,
        "y_all_raw": y_raw,
        "all_indices": indices,
        "val_idx": val_idx,
        "target_dmin": float(y_dmin[0, 0]),
        "target_dmax": float(y_dmax[0, 0]),
        "char_times": cfg.char_times,
        "name": cfg.name,
    }
