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
    hidden_nodes: int = 40
    lb_output_dim: int = 3
    activation: str = "sigmoid"
    lr: float = 0.001
    weight_init_low: float = 0.0
    weight_init_high: float = 0.5
    epochs: int = 5000
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
