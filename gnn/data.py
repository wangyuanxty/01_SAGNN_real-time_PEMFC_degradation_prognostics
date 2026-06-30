"""数据加载与预处理模块。

流水线: 原始 xlsx/csv → 拼接 → 提取特征化时刻 → 归一化 → AGO → 划分。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from gnn.config import DatasetConfig, DATA_ROOT


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """清洗从欧式 CSV 读入的数字列: 空格小数点 → 句点小数点, 转为 float。

    FC2 Ageing Part 2 CSV 格式: 分号分隔, 空格作小数点。
    例: "3 179" → 3.179, "70 066" → 70.066。
    """
    import re
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].astype(str).str.strip()
            # 连续空格 → 单个句点
            cleaned = cleaned.str.replace(r"\s+", ".", regex=True)
            # 残留逗号 → 句点 (如果存在)
            cleaned = cleaned.str.replace(",", ".")
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def load_raw_data(cfg: DatasetConfig) -> pd.DataFrame:
    """加载并纵向拼接一个数据集的所有 Ageing 文件。

    Args:
        cfg: 数据集配置。

    Returns:
        拼接后的 DataFrame, 保留原始列名。

    Raises:
        FileNotFoundError: 任一文件不存在。
    """
    source_dir = DATA_ROOT / cfg.source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {source_dir}")

    frames = []
    for fname in cfg.file_parts:
        filepath = source_dir / fname
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, sep=";", encoding="utf-8")
        else:
            df = pd.read_excel(filepath)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # 清洗数字列（CSV 可能残留空格）
    combined = _clean_numeric(combined)

    return combined


def extract_char_points(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> pd.DataFrame:
    """提取特征化时刻对应的数据行。

    对 cfg.char_times 中每个时间 t, 在 df["Time (h)"] 列中找最接近的行。

    Args:
        df: load_raw_data 返回的完整 DataFrame。
        cfg: 数据集配置。

    Returns:
        包含 8 行的 DataFrame, 列与 df 相同。

    Raises:
        KeyError: df 中缺少 "Time (h)" 列。
    """
    time_col = "Time (h)"
    if time_col not in df.columns:
        raise KeyError(f"数据中缺少时间列: '{time_col}'。可用列: {list(df.columns)}")

    rows = []
    for t in cfg.char_times:
        idx = (df[time_col] - t).abs().idxmin()
        rows.append(df.loc[idx])

    result = pd.DataFrame(rows, columns=df.columns)
    return result.reset_index(drop=True)


def build_samples(
    df: pd.DataFrame,
    cfg: DatasetConfig,
) -> "tuple[np.ndarray, np.ndarray]":
    """从特征化点 DataFrame 构建特征矩阵和目标向量。

    Args:
        df: extract_char_points 返回的 8 行 DataFrame。
        cfg: 数据集配置。

    Returns:
        X: (8, 5) 特征矩阵, float64。
        y: (8,) 目标向量, float64。

    Raises:
        ValueError: 特征化点数不是 8。
    """
    n_expected = len(cfg.char_times)
    if len(df) != n_expected:
        raise ValueError(
            f"特征化点数量不匹配: 期望 {n_expected}, 实际 {len(df)}"
        )

    feature_cols = list(cfg.feature_cols)
    target_col = cfg.target_col

    # 检查列是否存在
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise KeyError(f"缺少特征列: {missing_features}. 可用列: {list(df.columns)}")

    if target_col not in df.columns:
        raise KeyError(f"缺少目标列: '{target_col}'. 可用列: {list(df.columns)}")

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    return X, y


def normalize(data: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Min-max 归一化到 [0, 1]。

    对 2D 数据按列独立归一化, 对 1D 数据按整体归一化。

    Args:
        data: (n,) 或 (n, d) 数组。

    Returns:
        normalized: 归一化后的数组, shape 与输入相同。
        dmin: 最小值, shape 与 data[0] 相同。
        dmax: 最大值, shape 与 data[0] 相同。
    """
    data = np.asarray(data, dtype=np.float64)
    dmin = data.min(axis=0, keepdims=True) if data.ndim == 2 else np.array([data.min()])
    dmax = data.max(axis=0, keepdims=True) if data.ndim == 2 else np.array([data.max()])
    denom = dmax - dmin
    denom[denom == 0.0] = 1.0  # 防止常数列除零
    normalized = (data - dmin) / denom
    if data.ndim == 1:
        normalized = normalized.ravel()
    dmin = dmin.ravel()
    dmax = dmax.ravel()
    return normalized, dmin, dmax


def inverse_normalize(
    normalized: np.ndarray,
    dmin: np.ndarray,
    dmax: np.ndarray,
) -> np.ndarray:
    """逆归一化: 将 [0,1] 数据还原到原始尺度。

    Args:
        normalized: 归一化后的数据。
        dmin: normalize 返回的最小值。
        dmax: normalize 返回的最大值。

    Returns:
        原始尺度的数据。
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    dmin = np.asarray(dmin, dtype=np.float64)
    dmax = np.asarray(dmax, dtype=np.float64)
    return normalized * (dmax - dmin) + dmin


def ago_sequence(y: np.ndarray) -> np.ndarray:
    """一阶累加生成 (1-AGO), 论文公式 (2)。

    x^(1)(k) = Σ_{i=1}^{k} x^(0)(i)

    Args:
        y: (n,) 归一化后的目标序列。

    Returns:
        (n,) AGO 序列。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    return np.cumsum(y)


def iago_sequence(y_ago: np.ndarray) -> np.ndarray:
    """逆累加生成 (IAGO), 论文公式 (8)。

    x^(0)(k) = x^(1)(k) - x^(1)(k-1)

    Args:
        y_ago: (n,) AGO 预测序列。

    Returns:
        (n-1,) 还原后的序列。
    """
    y_ago = np.asarray(y_ago, dtype=np.float64).ravel()
    return np.diff(y_ago)


def load_dataset(cfg: DatasetConfig, seed: int = 42) -> dict:
    """完整数据加载流水线。

    1. load_raw_data → 2. extract_char_points → 3. build_samples
    → 4. normalize X → 5. normalize y → 6. AGO y
    → 7. 6/2 时间顺序划分 (前6训练, 后2验证)

    Args:
        cfg: 数据集配置。
        seed: 保留参数, 当前未使用 (时间顺序划分不需要随机种子)。

    Returns:
        dict with keys:
            X_train (6,5), X_val (2,5),
            y_train_ago (6,), y_val_ago (2,),
            y_train_raw (6,), y_val_raw (2,),
            X_all (8,5), y_all_ago (8,), y_all_raw (8,),
            X_dmin (5,), X_dmax (5,),
            target_dmin (float), target_dmax (float),
            char_times (tuple), name (str),
    """
    df = load_raw_data(cfg)
    char_df = extract_char_points(df, cfg)
    X, y_raw = build_samples(char_df, cfg)

    # 归一化特征（按列）
    X_norm, X_dmin, X_dmax = normalize(X)  # X_dmin/dmax: (5,)

    # 归一化目标（标量）
    y_norm, y_dmin, y_dmax = normalize(y_raw)  # 1D

    # AGO
    y_ago = ago_sequence(y_norm)

    # 6/2 时间顺序划分 (前6训练, 后2验证)
    # 退化预测是外推任务: 必须用过去预测未来
    train_idx = np.arange(6)   # 前 6 个特征化时刻
    val_idx = np.arange(6, 8)  # 后 2 个特征化时刻

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
        "X_dmin": X_dmin,
        "X_dmax": X_dmax,
        "target_dmin": float(y_dmin[0]),
        "target_dmax": float(y_dmax[0]),
        "char_times": cfg.char_times,
        "name": cfg.name,
    }
