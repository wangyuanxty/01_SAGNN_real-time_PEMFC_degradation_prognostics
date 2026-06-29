"""GNN 超参数配置（来自论文 Section 3.3）。"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "FC1_FC2_Excel"


@dataclass(frozen=True)
class DatasetConfig:
    """单个数据集的配置。"""
    name: str                                    # "FC1" 或 "FC2"
    source_dir: str                              # 数据子目录名
    file_parts: list[str]                        # Ageing 文件名列表
    char_times: list[float]                      # 8 个特征化时刻 (小时)
    feature_cols: list[str]                      # 5 个输入特征列名
    target_col: str                              # 预测目标列名

    @property
    def csv_sep(self) -> str:
        """FC2 part2 是分号分隔的 CSV，其余是 Excel。"""
        return ";"

    @property
    def csv_decimal(self) -> str:
        """欧式数字格式：逗号小数点。"""
        return ","


@dataclass(frozen=True)
class TrainConfig:
    """训练超参数。"""
    hidden_nodes: int = 10
    activation: str = "sigmoid"
    optimizer: str = "adam"
    learning_rate: float = 0.0002
    weight_init_low: float = 0.0
    weight_init_high: float = 0.5
    epochs: int = 1000
    loss_fn: str = "mse"
    train_points: int = 6
    val_points: int = 2
    random_seed: int = 42


FEATURE_COLS = [
    "Utot (V)",
    "HrAIRFC (%)",
    "PinH2 (mbara)",
    "TinWAT (ｰC)",
    "I (A)",
]
TARGET_COL = "Utot (V)"

FC1_CONFIG = DatasetConfig(
    name="FC1",
    source_dir="FC1_Without_Ripples_Excel",
    file_parts=[
        "FC1_Ageing_part1.xlsx",
        "FC1_Ageing_part2.xlsx",
        "FC1_Ageing_part3.xlsx",
    ],
    char_times=[0.0, 48.0, 185.0, 348.0, 515.0, 658.0, 823.0, 991.0],
    feature_cols=FEATURE_COLS,
    target_col=TARGET_COL,
)

FC2_CONFIG = DatasetConfig(
    name="FC2",
    source_dir="FC2_With_Ripples_Excel",
    file_parts=[
        "FC2_Ageing_part1.xlsx",
        "FC2_Ageing_part2.csv",
    ],
    char_times=[0.0, 35.0, 182.0, 343.0, 515.0, 666.0, 830.0, 1016.0],
    feature_cols=FEATURE_COLS,
    target_col=TARGET_COL,
)

TRAIN_CONFIG = TrainConfig()

DATASETS = [FC1_CONFIG, FC2_CONFIG]
