"""标准灰色神经网络 (GNN) 模型 — 论文原始架构。

结构严格对应论文公式 (4) 和公式 (6):

  x̂^(1)(k+1) = (x^(0)(1) − drive) · e^(−a·k) + drive                  (4)
  a = −w11,   b_i = w_{1,i+1} / w11   (i=1..5)                        (6)

w11:    标量参数 → a = −w11, 指数衰减核 e^(−a·k)
drive:  Linear(5→1), 特征 → 驱动项 (训练时不除 w11)
灰组合: (x0 − drive) · e^(−a·k) + drive
hidden: Linear(1→hidden) + Sigmoid
output: Linear(hidden→1)
"""

import numpy as np
import torch
import torch.nn as nn


class GNN(nn.Module):
    """标准灰色神经网络。

    Args:
        input_dim:  特征维度, 默认 5。
        hidden_dim: 隐藏层节点数, 默认 40。
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 40):
        super().__init__()

        # 衰减系数: w11 → a = −w11 > 0, e^(−a·k)
        self.w11 = nn.Parameter(torch.tensor(-0.2))

        # 特征编码器: u(5) → 驱动项标量
        self.drive = nn.Linear(input_dim, 1, bias=False)

        # 隐藏层 + 输出层
        self.hidden = nn.Linear(1, hidden_dim, bias=True)
        self.output = nn.Linear(hidden_dim, 1, bias=True)
        self._init_layers()

    def _init_layers(self) -> None:
        for m in [self.hidden, self.output]:
            nn.init.uniform_(m.weight, 0.0, 0.5)
            if m.bias is not None:
                nn.init.uniform_(m.bias, 0.0, 0.5)
        nn.init.uniform_(self.drive.weight, -0.25, 0.25)

    # ── 前向传播 ────────────────────────────────────────────────

    def forward(
        self, k: torch.Tensor, u: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """训练时前向传播。

        Args:
            k:  (batch,)  时间步索引。
            u:  (batch, 5) 归一化特征。
            x0: scalar,   x^(0)(1)。
        Returns:
            (batch,) AGO 空间预测值。
        """
        a = -self.w11                              # a = −w11  (公式 6)
        decay = torch.exp(-a * k)                  # 指数衰减核 e^(−a·k)
        drive = self.drive(u).squeeze(-1)          # 特征 → 驱动项
        grey = (x0 - drive) * decay + drive         # 灰组合 (公式 4)
        h = torch.sigmoid(self.hidden(grey.unsqueeze(-1)))
        return self.output(h).squeeze(-1)

    # ── 灰系数提取 ──────────────────────────────────────────────

    def extract_grey_coeffs(self) -> "tuple[float, np.ndarray]":
        """从权重提取灰系数 (公式 6)。"""
        w11 = float(self.w11.item())
        a = -w11
        w = self.drive.weight.data[0].cpu().numpy()
        b = w / w11
        return a, b

    # ── 解析预测 ────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        """灰系数 → 时间响应函数 (公式 4) → IAGO → 逆归一化。"""
        a, bs = self.extract_grey_coeffs()

        X_all = data["X_all"]
        n = X_all.shape[0]
        x0 = float(X_all[0, 0])

        y_ago_pred = np.zeros(n, dtype=np.float64)
        y_ago_pred[0] = x0

        for i in range(1, n):
            u = X_all[i]
            steady = float(np.dot(bs, u)) / a
            y_ago_pred[i] = (x0 - steady) * np.exp(-a * i) + steady

        y_pred_norm = np.diff(y_ago_pred)
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
