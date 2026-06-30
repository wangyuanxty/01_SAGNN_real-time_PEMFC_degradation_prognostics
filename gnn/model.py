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
