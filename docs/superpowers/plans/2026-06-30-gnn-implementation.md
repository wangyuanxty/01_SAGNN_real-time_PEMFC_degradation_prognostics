# GNN (标准灰色神经网络) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 PyTorch 实现论文的 4 层标准 Grey Neural Network（LA→LB→LC→LD），包含完整数据加载、训练和评估流水线。

**Architecture:** 6 文件模块化设计。config 定义配置和数据常量，data 负责原始文件加载→特征化点提取→归一化→AGO→划分，model 实现 4 层 GNN + 灰系数提取 + 解析预测，train 执行训练循环，main 提供端到端入口和可视化。测试覆盖数据变换的数学回环和集成流水线。

**Tech Stack:** Python 3.10+, PyTorch 2.x, pandas, openpyxl, matplotlib, pytest

## Global Constraints

- Python ≥ 3.10
- PyTorch ≥ 2.0
- 数据目录: `FC1_FC2_Excel/` 相对于项目根目录
- 特征列: `Utot (V)`, `HrAIRFC (%)`, `PinH2 (mbara)`, `TinWAT (C)`, `I (A)`
- 目标列: `Utot (V)`（自回归）
- LB 维度 = 3, LC 隐藏节点 = 10
- 优化器: Adam, lr=0.001
- 权重初始化: U(0, 0.5)
- 最大 epoch: 1000
- 训练/验证: 6/2（8 个特征化点）
- 随机种子: 42（可覆盖）
- 列名精确匹配：FC1/FC2 Ageing 文件首行 header，含全角字符 `ｰ`（U+FF70）替代 `°`, 上标 `ｲ`（U+FF72）替代 `²`
- 欧式 CSV: 分号分隔 (`sep=";"`), 空格+逗号小数点 (`decimal=","`), 数字内部含空格需清洗

---

### Task 1: 配置模块 `gnn/config.py`

**Files:**
- Create: `gnn/__init__.py`
- Create: `gnn/config.py`

**Interfaces:**
- Produces: `DatasetConfig` (frozen dataclass), `TrainConfig` (frozen dataclass), `FEATURE_COLS: list[str]`, `TARGET_COL: str`, `FC1_CONFIG: DatasetConfig`, `FC2_CONFIG: DatasetConfig`, `TRAIN_CONFIG: TrainConfig`, `DATASETS: list[DatasetConfig]`

- [ ] **Step 1: 创建 `gnn/__init__.py`**

```python
"""GNN: Grey Neural Network for PEMFC degradation prognostics."""
```

- [ ] **Step 2: 创建 `gnn/config.py`**

```python
"""GNN 配置 —— 数据集定义与训练超参数。

论文参考: Sapnken et al. (2025) §3.3 Parametric settings.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "FC1_FC2_Excel"

FEATURE_COLS = [
    "Utot (V)",
    "HrAIRFC (%)",
    "PinH2 (mbara)",
    "TinWAT (ｰC)",
    "I (A)",
]
TARGET_COL = "Utot (V)"


@dataclass(frozen=True)
class DatasetConfig:
    """单个 PEMFC 退化数据集的元信息。

    Attributes:
        name: 数据集短名, "FC1" 或 "FC2"。
        source_dir: FC1_FC2_Excel/ 下的子目录名。
        file_parts: Ageing 文件名列表（按时间顺序拼接）。
        char_times: 8 个特征化时刻 (小时), 对应论文 Table 1/2。
        feature_cols: 5 个输入特征列名（Excel 中的精确 header）。
        target_col: 预测目标列名（与 feature_cols 的第一个相同, 自回归）。
    """
    name: str
    source_dir: str
    file_parts: tuple[str, ...]
    char_times: tuple[float, ...]
    feature_cols: tuple[str, ...]
    target_col: str


@dataclass(frozen=True)
class TrainConfig:
    """GNN 训练超参数。

    默认值来自论文 §3.3, 仅将 SGD 替换为 Adam。
    """
    hidden_nodes: int = 10
    lb_output_dim: int = 3
    activation: str = "sigmoid"
    lr: float = 0.001
    weight_init_low: float = 0.0
    weight_init_high: float = 0.5
    epochs: int = 1000
    train_points: int = 6
    val_points: int = 2
    seed: int = 42


FC1_CONFIG = DatasetConfig(
    name="FC1",
    source_dir="FC1_Without_Ripples_Excel",
    file_parts=(
        "FC1_Ageing_part1.xlsx",
        "FC1_Ageing_part2.xlsx",
        "FC1_Ageing_part3.xlsx",
    ),
    char_times=(0.0, 48.0, 185.0, 348.0, 515.0, 658.0, 823.0, 991.0),
    feature_cols=tuple(FEATURE_COLS),
    target_col=TARGET_COL,
)

FC2_CONFIG = DatasetConfig(
    name="FC2",
    source_dir="FC2_With_Ripples_Excel",
    file_parts=(
        "FC2_Ageing_part1.xlsx",
        "FC2_Ageing_part2.csv",
    ),
    char_times=(0.0, 35.0, 182.0, 343.0, 515.0, 666.0, 830.0, 1016.0),
    feature_cols=tuple(FEATURE_COLS),
    target_col=TARGET_COL,
)

TRAIN_CONFIG = TrainConfig()

DATASETS = [FC1_CONFIG, FC2_CONFIG]
```

**注意:** 列名含全角字符 `ｰ`（ｰ）和 `ｲ`（ｲ）。这是 Excel 文件中的实际编码，必须精确匹配。`FEATURE_COLS` 中的 `TinWAT (ｰC)` 对应 header 的 `TinWAT (ｰC)`（全角度符号），`I (A)` 对应 `I (A)`。

- [ ] **Step 3: 验证 config 可导入**

```powershell
python -c "from gnn.config import FC1_CONFIG, FC2_CONFIG, TRAIN_CONFIG; print('FC1:', FC1_CONFIG.name, '| features:', len(FC1_CONFIG.feature_cols), '| char_times:', len(FC1_CONFIG.char_times)); print('FC2:', FC2_CONFIG.name, '| features:', len(FC2_CONFIG.feature_cols), '| char_times:', len(FC2_CONFIG.char_times)); print('Train epochs:', TRAIN_CONFIG.epochs)"
```

Expected output:
```
FC1: FC1 | features: 5 | char_times: 8
FC2: FC2 | features: 5 | char_times: 8
Train epochs: 1000
```

- [ ] **Step 4: Commit**

```powershell
git add gnn/__init__.py gnn/config.py
git commit -m "feat: add gnn/config.py with DatasetConfig and TrainConfig"
```

---

### Task 2: 数据加载与预处理 `gnn/data.py`

**Files:**
- Create: `gnn/data.py`

**Interfaces:**
- Consumes: `gnn.config.DatasetConfig`, `gnn.config.DATA_ROOT`
- Produces:
  - `load_raw_data(cfg: DatasetConfig) -> pd.DataFrame`
  - `extract_char_points(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame`
  - `build_samples(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray]`
  - `normalize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `inverse_normalize(norm: np.ndarray, dmin: np.ndarray, dmax: np.ndarray) -> np.ndarray`
  - `ago_sequence(y: np.ndarray) -> np.ndarray`
  - `iago_sequence(y_ago: np.ndarray) -> np.ndarray`
  - `load_dataset(cfg: DatasetConfig, seed: int = 42) -> dict`

- [ ] **Step 1: 创建 `gnn/data.py`**

```python
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
    → 7. 6/2 随机划分

    Args:
        cfg: 数据集配置。
        seed: 随机种子。

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
    rng = np.random.default_rng(seed)

    df = load_raw_data(cfg)
    char_df = extract_char_points(df, cfg)
    X, y_raw = build_samples(char_df, cfg)

    # 归一化特征（按列）
    X_norm, X_dmin, X_dmax = normalize(X)  # X_dmin/dmax: (5,)

    # 归一化目标（标量）
    y_norm, y_dmin, y_dmax = normalize(y_raw)  # 1D

    # AGO
    y_ago = ago_sequence(y_norm)

    # 6/2 随机划分
    indices = np.arange(8)
    rng.shuffle(indices)
    train_idx = np.sort(indices[:6])
    val_idx = np.sort(indices[6:])

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
```

- [ ] **Step 2: 验证数据流水线形状和数学回环**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from gnn.config import FC1_CONFIG, FC2_CONFIG
from gnn.data import (
    load_raw_data, extract_char_points, build_samples,
    normalize, inverse_normalize, ago_sequence, iago_sequence,
    load_dataset,
)

# 1. normalize 回环
x = np.array([[3.3, 50.0, 1300.0, 55.0, 70.0],
              [3.2, 52.0, 1290.0, 54.0, 69.0]], dtype=np.float64)
x_norm, dmin, dmax = normalize(x)
x_back = inverse_normalize(x_norm, dmin, dmax)
assert np.allclose(x, x_back), f'normalize roundtrip failed: max_diff={np.abs(x-x_back).max()}'
print('[PASS] normalize roundtrip')

# 2. AGO/IAGO 回环
y = np.array([3.416, 3.397, 3.360, 3.342, 3.322, 3.298, 3.265, 3.270])
y_ago = ago_sequence(y)
y_iago = iago_sequence(y_ago)
assert len(y_iago) == 7, f'expected 7, got {len(y_iago)}'
assert np.allclose(y[1:], y_iago), f'AGO/IAGO roundtrip failed'
print('[PASS] AGO/IAGO roundtrip')

# 3. FC1 数据流水线
data = load_dataset(FC1_CONFIG, seed=42)
assert data['X_train'].shape == (6, 5), f'X_train shape: {data[\"X_train\"].shape}'
assert data['X_val'].shape == (2, 5), f'X_val shape: {data[\"X_val\"].shape}'
assert data['y_train_ago'].shape == (6,), f'y_train_ago shape: {data[\"y_train_ago\"].shape}'
assert data['X_all'].shape == (8, 5), f'X_all shape: {data[\"X_all\"].shape}'
assert data['y_all_raw'].shape == (8,), f'y_all_raw shape: {data[\"y_all_raw\"].shape}'
print(f'[PASS] FC1 pipeline: X_train={data[\"X_train\"].shape}, y_train_ago={data[\"y_train_ago\"].shape}')
print(f'  target_dmin={data[\"target_dmin\"]:.4f}, target_dmax={data[\"target_dmax\"]:.4f}')
print(f'  y_all_raw = {data[\"y_all_raw\"]}')

# 4. FC2 数据流水线
data2 = load_dataset(FC2_CONFIG, seed=42)
assert data2['X_train'].shape == (6, 5)
print(f'[PASS] FC2 pipeline: X_train={data2[\"X_train\"].shape}')
print(f'  y_all_raw = {data2[\"y_all_raw\"]}')
"
```

Expected: 4 个 `[PASS]`，无 assert 错误。

- [ ] **Step 3: Commit**

```powershell
git add gnn/data.py
git commit -m "feat: add gnn/data.py with data loading and preprocessing"
```

---

### Task 3: GNN 模型 `gnn/model.py`

**Files:**
- Create: `gnn/model.py`

**Interfaces:**
- Consumes: (nothing from earlier tasks except general PyTorch)
- Produces:
  - `class GNN(nn.Module)` — `forward(x) -> Tensor`, `extract_grey_coeffs() -> tuple[float, np.ndarray]`, `predict(data: dict) -> dict`
  - `def _init_weights(module)` — helper for U(0, 0.5) init

- [ ] **Step 1: 创建 `gnn/model.py`**

```python
"""标准灰色神经网络 (GNN) 模型。

4 层结构: LA (Identity) → LB (Linear, 灰化) → LC (Sigmoid) → LD (Linear)。

训练时 forward 输出 AGO 空间预测值。
训练后 extract_grey_coeffs() 提取灰系数 a, b_i,
predict() 走时间响应函数解析预测 + IAGO + 逆归一化。
"""

import numpy as np
import torch
import torch.nn as nn


class GNN(nn.Module):
    """标准 4 层灰色神经网络。

    LA: Identity (5 → 5)
    LB: Linear(5, 3, bias=True)  — 灰化层, 第一个神经元编码 a, b_i
    LC: Linear(3, 10) + Sigmoid   — 非线性隐藏层
    LD: Linear(10, 1)             — 输出层

    Args:
        input_dim: 输入特征维度, 默认 5。
        lb_dim: LB 层输出维度, 默认 3。
        hidden_dim: LC 层隐藏节点数, 默认 10。
        init_low, init_high: 权重均匀初始化范围, 默认 (0, 0.5)。
    """

    def __init__(
        self,
        input_dim: int = 5,
        lb_dim: int = 3,
        hidden_dim: int = 10,
        init_low: float = 0.0,
        init_high: float = 0.5,
    ):
        super().__init__()
        self.LA = nn.Identity()
        self.LB = nn.Linear(input_dim, lb_dim, bias=True)
        self.LC = nn.Linear(lb_dim, hidden_dim, bias=True)
        self.LD = nn.Linear(hidden_dim, 1, bias=True)

        # 自定义权重初始化
        self.apply(lambda m: _init_weights(m, init_low, init_high))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """训练时前向传播。

        Args:
            x: (batch, 5) 归一化特征。

        Returns:
            (batch,) AGO 空间预测值。
        """
        x = self.LA(x)                 # (batch, 5)
        x = self.LB(x)                 # (batch, 3)
        x = torch.sigmoid(self.LC(x))  # (batch, 10)
        x = self.LD(x)                 # (batch, 1)
        return x.squeeze(-1)           # (batch,)

    # ── 灰系数提取 ────────────────────────────────────────────

    def extract_grey_coeffs(self) -> "tuple[float, np.ndarray]":
        """从 LB 层第一个神经元的权重和偏置提取灰系数。

        映射关系 (论文公式 6):
            a  = -w11
            b1 = w12 / w11, b2 = w13 / w11, b3 = w14 / w11, b4 = w15 / w11
            b5 = bias / w11

        Returns:
            a: float, 发展系数。
            bs: (5,) ndarray, 驱动系数 b1..b5。

        Raises:
            RuntimeError: w11 ≈ 0, 灰模型奇点。
        """
        w = self.LB.weight.data[0].cpu().numpy()  # (5,)
        b = self.LB.bias.data[0].cpu().item()     # scalar
        w11 = float(w[0])

        if abs(w11) < 1e-10:
            raise RuntimeError(
                f"w11 ≈ 0 ({w11:.2e}), 灰模型退化。尝试不同随机种子。"
            )

        a = -w11
        bs = np.array([
            w[1] / w11,  # b1: 电压
            w[2] / w11,  # b2: 湿度
            w[3] / w11,  # b3: 氢压
            w[4] / w11,  # b4: 温度
            b / w11,     # b5: 电流 (由 bias 编码)
        ], dtype=np.float64)

        return a, bs

    # ── 解析预测 (不走 forward) ──────────────────────────────────

    def predict(self, data: dict) -> dict:
        """训练后使用时间响应函数做解析预测。

        流程: 提取 a, b_i → 对每个 k 计算时间响应 → IAGO → 逆归一化。

        Args:
            data: load_dataset 返回的字典, 需要 X_all, target_dmin,
                  target_dmax, y_all_raw, char_times。

        Returns:
            dict with keys:
                y_pred: (7,) ndarray, 预测电压 (k=1..7)。
                y_true: (7,) ndarray, 真实电压 (k=1..7)。
                a: float, 提取的发展系数。
                bs: (5,) ndarray, 提取的驱动系数。
        """
        a, bs = self.extract_grey_coeffs()

        X_all = data["X_all"]          # (8, 5)
        char_times = data["char_times"]  # (8,)
        n = len(char_times)

        x0 = float(X_all[0, 0])        # x^(0)(1): 第一个点的归一化电压

        # 时间响应函数 (公式 4):
        #   x̂^(1)(k+1) = (x0 - Σb_i*u_i / a) * exp(-a*k) + Σb_i*u_i / a
        k_vals = np.arange(n, dtype=np.float64)
        y_ago_pred = np.zeros(n, dtype=np.float64)
        y_ago_pred[0] = x0

        for i in range(1, n):
            u = X_all[i]               # 当前点的 5 个特征
            u_sum = float(np.dot(bs, u))
            k = k_vals[i]
            y_ago_pred[i] = (x0 - u_sum / a) * np.exp(-a * k) + u_sum / a

        # IAGO (公式 8) → 得到 k=1..7 的预测
        y_pred_norm = np.diff(y_ago_pred)  # (7,)

        # 逆归一化
        y_pred = (
            y_pred_norm * (data["target_dmax"] - data["target_dmin"])
            + data["target_dmin"]
        )

        return {
            "y_pred": y_pred,
            "y_true": data["y_all_raw"][1:],
            "a": a,
            "bs": bs,
        }


def _init_weights(module: nn.Module, low: float, high: float) -> None:
    """对 Linear 层做 U(low, high) 权重和偏置初始化。"""
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, low, high)
        if module.bias is not None:
            nn.init.uniform_(module.bias, low, high)
```

- [ ] **Step 2: 单元测试 —— 灰系数提取**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
import torch
from gnn.model import GNN

model = GNN()
model.eval()

# 手动设置 LB 权重
with torch.no_grad():
    model.LB.weight.copy_(torch.tensor([
        [-0.3, 0.06, 0.09, 0.12, 0.15],  # w11=-0.3 → a=0.3, b1=-0.2, b2=-0.3, b3=-0.4, b4=-0.5
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.1, 0.4, 0.3, 0.5],
    ], dtype=torch.float32))
    model.LB.bias.copy_(torch.tensor([0.045, 0.1, 0.2]))  # b5 = 0.045 / -0.3 = -0.15
    # 固定 LC, LD 权重避免干扰
    model.LC.weight.data.fill_(0.0)
    model.LC.bias.data.fill_(0.0)
    model.LD.weight.data.fill_(0.0)
    model.LD.bias.data.fill_(0.0)

a, bs = model.extract_grey_coeffs()
print(f'a  = {a:.6f}   (expected 0.3)')
print(f'b1 = {bs[0]:.6f} (expected -0.2)')
print(f'b2 = {bs[1]:.6f} (expected -0.3)')
print(f'b3 = {bs[2]:.6f} (expected -0.4)')
print(f'b4 = {bs[3]:.6f} (expected -0.5)')
print(f'b5 = {bs[4]:.6f} (expected -0.15)')

assert abs(a - 0.3) < 1e-6, f'a={a}'
assert abs(bs[0] - (-0.2)) < 1e-6, f'b1={bs[0]}'
assert abs(bs[1] - (-0.3)) < 1e-6, f'b2={bs[1]}'
assert abs(bs[2] - (-0.4)) < 1e-6, f'b3={bs[2]}'
assert abs(bs[3] - (-0.5)) < 1e-6, f'b4={bs[3]}'
assert abs(bs[4] - (-0.15)) < 1e-6, f'b5={bs[4]}'
print('[PASS] grey coefficient extraction')
"
```

Expected: `[PASS] grey coefficient extraction`

- [ ] **Step 3: 单元测试 —— forward 输出形状**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch
from gnn.model import GNN

model = GNN()
model.train()
x = torch.randn(6, 5)
y = model(x)
assert y.shape == (6,), f'forward output shape: {y.shape}, expected (6,)'
print(f'[PASS] forward output shape: {y.shape}')

# 单样本也 OK
y1 = model(x[:1])
assert y1.shape == (1,), f'single sample shape: {y1.shape}'
print(f'[PASS] single sample shape: {y1.shape}')
"
```

Expected: `[PASS] forward output shape` × 2

- [ ] **Step 4: Commit**

```powershell
git add gnn/model.py
git commit -m "feat: add gnn/model.py with GNN class and grey coefficient extraction"
```

---

### Task 4: 训练循环 `gnn/train.py`

**Files:**
- Create: `gnn/train.py`

**Interfaces:**
- Consumes: `gnn.model.GNN`, `gnn.config.TrainConfig`
- Produces: `train(model: GNN, data: dict, config: TrainConfig) -> tuple[GNN, list[float]]`

- [ ] **Step 1: 创建 `gnn/train.py`**

```python
"""GNN 训练循环。

小样本 (6 个训练点) → 全批量梯度下降, Adam 优化器。
"""

import torch
import torch.nn as nn
from gnn.model import GNN
from gnn.config import TrainConfig


def train(
    model: GNN,
    data: dict,
    config: TrainConfig,
    verbose: bool = True,
) -> "tuple[GNN, list[float]]":
    """训练 GNN 模型。

    Args:
        model: 未训练的 GNN 实例。
        data: load_dataset 返回的字典。
        config: 训练超参数。
        verbose: 是否每 100 epoch 打印 loss。

    Returns:
        model: 训练后的 GNN (in-place 修改 + 返回)。
        losses: 每个 epoch 的训练 loss 列表。
    """
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train_ago"], dtype=torch.float32)

    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}/{config.epochs}  loss = {loss.item():.8f}")

    if verbose:
        print(f"  epoch {config.epochs:4d}/{config.epochs}  loss = {losses[-1]:.8f}  [done]")

    return model, losses
```

- [ ] **Step 2: 验证训练可运行**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch
from gnn.config import FC1_CONFIG, TRAIN_CONFIG
from gnn.data import load_dataset
from gnn.model import GNN
from gnn.train import train

torch.manual_seed(42)
data = load_dataset(FC1_CONFIG, seed=42)
model = GNN()

model, losses = train(model, data, TRAIN_CONFIG)
assert len(losses) == TRAIN_CONFIG.epochs, f'expected {TRAIN_CONFIG.epochs} losses, got {len(losses)}'
assert losses[-1] < losses[0] or losses[0] < 0.01, f'loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}'
print(f'[PASS] training ran {len(losses)} epochs, loss: {losses[0]:.6f} -> {losses[-1]:.8f}')
"
```

Expected: `[PASS] training ran 1000 epochs`，loss 显著下降。

- [ ] **Step 3: Commit**

```powershell
git add gnn/train.py
git commit -m "feat: add gnn/train.py with full-batch Adam training loop"
```

---

### Task 5: 评估与可视化 `gnn/main.py`

**Files:**
- Create: `gnn/main.py`

**Interfaces:**
- Consumes: `gnn.model.GNN`, `gnn.train.train`, `gnn.data.load_dataset`, `gnn.config.{FC1_CONFIG, FC2_CONFIG, TRAIN_CONFIG}`
- Produces: `evaluate(model: GNN, data: dict) -> dict`, `plot_degradation(results: dict, data: dict, save_path: str|None)`, `plot_loss(losses: list[float], config: TrainConfig, save_path: str|None)`, `main(dataset_name: str, seed: int) -> dict`

- [ ] **Step 1: 创建 `gnn/main.py`**

```python
"""端到端入口: 数据 → 训练 → 评估 → 可视化。"""

import numpy as np
import matplotlib.pyplot as plt
from gnn.config import FC1_CONFIG, FC2_CONFIG, TRAIN_CONFIG, TrainConfig
from gnn.data import load_dataset
from gnn.model import GNN
from gnn.train import train


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
    y_pred = pred["y_pred"]
    y_true = pred["y_true"]

    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
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
    plt.show()


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
    plt.show()


def print_metrics(results: dict) -> None:
    """打印评估指标和灰系数。"""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  MAPE : {results['MAPE']:.4f} %")
    print(f"  MSE  : {results['MSE']:.8f} V²")
    print(f"  RMSE : {results['RMSE']:.8f} V")
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
    for i in range(len(results["y_pred"])):
        print(
            f"  point {i+1}:  pred={results['y_pred'][i]:.6f} V  "
            f"actual={results['y_true'][i]:.6f} V  "
            f"error={abs(results['y_pred'][i]-results['y_true'][i]):.6f} V"
        )


def main(dataset_name: str = "FC1", seed: int = 42) -> dict:
    """端到端入口。

    Args:
        dataset_name: "FC1" 或 "FC2"。
        seed: 随机种子。

    Returns:
        results 字典。
    """
    import torch
    torch.manual_seed(seed)

    cfg = FC1_CONFIG if dataset_name.upper() == "FC1" else FC2_CONFIG
    train_cfg = TRAIN_CONFIG

    print(f"\n{'='*60}")
    print(f"  SiGDSM-GNN — Standard Grey Neural Network")
    print(f"  Dataset: {cfg.name}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    # 1. 数据
    print("\n[1/4] Loading data...")
    data = load_dataset(cfg, seed=seed)
    print(f"  Train points: {len(data['y_train_ago'])}, Val points: {len(data['y_val_ago'])}")

    # 2. 模型
    print("\n[2/4] Building model...")
    model = GNN(
        input_dim=5,
        lb_dim=train_cfg.lb_output_dim,
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
```

- [ ] **Step 2: 集成测试 —— 端到端运行 FC1**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch; torch.manual_seed(42)
import matplotlib; matplotlib.use('Agg')  # 无头模式
from gnn.main import main

results = main('FC1', seed=42)
assert results['MAPE'] < 10.0, f'MAPE too high: {results[\"MAPE\"]:.2f}%'
print(f'[PASS] FC1 pipeline: MAPE={results[\"MAPE\"]:.4f}%, MSE={results[\"MSE\"]:.8f}')
"
```

Expected: MAPE < 10%，流水线顺利跑通。

- [ ] **Step 3: 集成测试 —— 端到端运行 FC2**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch; torch.manual_seed(42)
import matplotlib; matplotlib.use('Agg')
from gnn.main import main

results = main('FC2', seed=42)
assert results['MAPE'] < 10.0, f'MAPE too high: {results[\"MAPE\"]:.2f}%'
print(f'[PASS] FC2 pipeline: MAPE={results[\"MAPE\"]:.4f}%, MSE={results[\"MSE\"]:.8f}')
"
```

- [ ] **Step 4: Commit**

```powershell
git add gnn/main.py
git commit -m "feat: add gnn/main.py with evaluation, metrics, and visualization"
```

---

### Task 6: 多 seed 统计与最终验证

**Files:**
- Modify: (none — execution-only step)

**Interfaces:**
- Consumes: `gnn.main.main`

- [ ] **Step 1: 多 seed 运行 FC1, 统计 MAPE 分布**

```powershell
python -c "
import sys; sys.path.insert(0, '.')
import torch
import matplotlib; matplotlib.use('Agg')
import numpy as np
from gnn.main import main

seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
mapes = []
mses = []
for s in seeds:
    torch.manual_seed(s)
    results = main('FC1', seed=s)
    mapes.append(results['MAPE'])
    mses.append(results['MSE'])
    print(f'  seed={s:4d}: MAPE={results[\"MAPE\"]:.4f}%')

print(f'\n=== Summary over {len(seeds)} seeds ===')
print(f'MAPE: mean={np.mean(mapes):.4f}%, std={np.std(mapes):.4f}%, min={np.min(mapes):.4f}%, max={np.max(mapes):.4f}%')
print(f'MSE:  mean={np.mean(mses):.8f}, std={np.std(mses):.8f}')
print(f'[PASS] Multi-seed evaluation complete')
"
```

- [ ] **Step 2: Commit final results**

```powershell
git add -A
git commit -m "chore: final verification with multi-seed statistics on FC1 and FC2"
```
