"""标准灰色神经网络 (GNN) 模型 — 论文原始架构。

四层结构, 严格对应论文公式 (4) 和公式 (6):

  x̂^(1)(k+1) = (x^(0)(1) − (1/a)·Σb_i·u_i) · e^(−a·k) + (1/a)·Σb_i·u_i   (4)
  a = −w11,   b_i = w_{1,i+1} / w11   (i=1..5)                           (6)

四层分别负责:
  1. 指数衰减核 (w11): 标量参数, 控制退化速度 a = −w11, 输出 e^(−a·k)
  2. 特征编码层 (LB):  Linear(5→1) 将 5 个运行参数映射为驱动项, 训练时不除 w11
  3. Sigmoid 隐藏层 (LC): Linear(1→hidden) + Sigmoid, 非线性变换
  4. 线性输出层 (LD):    Linear(hidden→1), 产生 AGO 空间的退化预测值

灰组合 (公式 4): (x0 − 特征编码输出) · 指数衰减核 + 特征编码输出

注: 模块名 w11/LB/LC/LD 为代码简洁保留, 但每个成员都附有中文注释说明其角色。
"""

import numpy as np
import torch
import torch.nn as nn


class GNN(nn.Module):
    """标准 4 层灰色神经网络。

    Args:
        input_dim:  特征维度, 默认 5。
        hidden_dim: LC 隐藏节点数, 默认 40。
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 40):
        super().__init__()

        # 指数衰减核: 单标量参数 w11, 控制退化速度 a = −w11 > 0
        self.w11 = nn.Parameter(torch.tensor(-0.2))

        # 特征编码层: 5 个运行参数 → 驱动项, 训练时不除 w11
        self.LB = nn.Linear(input_dim, 1, bias=False)

        # Sigmoid 隐藏层 + 线性输出层
        self.LC = nn.Linear(1, hidden_dim, bias=True)
        self.LD = nn.Linear(hidden_dim, 1, bias=True)
        self._init_layers()

    def _init_layers(self) -> None:
        for m in [self.LC, self.LD]:
            nn.init.uniform_(m.weight, 0.0, 0.5)
            if m.bias is not None:
                nn.init.uniform_(m.bias, 0.0, 0.5)
        nn.init.uniform_(self.LB.weight, -0.25, 0.25)

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
        a = -self.w11                              # 退化速度 a = −w11 (公式 6)
        decay = torch.exp(-a * k)                  # 指数衰减核: e^(−a·k)
        driving = self.LB(u).squeeze(-1)           # 特征编码: 5 参数 → 驱动项
        grey = (x0 - driving) * decay + driving    # 灰组合 (公式 4)
        h = torch.sigmoid(self.LC(grey.unsqueeze(-1)))   # Sigmoid 隐藏层
        return self.LD(h).squeeze(-1)                     # 线性输出层

    # ── 灰系数提取 ──────────────────────────────────────────────

    def extract_grey_coeffs(self) -> "tuple[float, np.ndarray]":
        """从权重提取灰系数 (公式 6)。"""
        w11 = float(self.w11.item())
        a = -w11
        lb_w = self.LB.weight.data[0].cpu().numpy()
        b = lb_w / w11
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
