"""GNN 深度化扩展 — 三个方向, 通过 DeepGNN 的 bool 参数任意组合。

  deep_lb:  深层特征编码器, 替代单层特征编码层 (MLP 5→16→8→1), 学习特征交互。
  deep_la:  时变衰减率 a(k), 替代常数退化速度, 适应不同退化阶段。
  residual: 显式灰趋势 + 深度残差, 灰模型负责不可逆退化骨架, 残差网络学习可逆波动。

继承基类 GNN, 因此 w11(指数衰减核)/LB(特征编码)/LC(隐藏层)/LD(输出层) 的含义与基类一致。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.model import GNN


class DeepGNN(GNN):
    """可组合的深度化 GNN, 三个 bool 参数任意排列。

    继承标准 GNN (论文原始架构), 通过三个开关控制深度化方向:
      deep_lb   — LB: Linear(5,1) → MLP(5→16→8→1)
      deep_la   — LA: 常数 a → a(k) = softplus(−w11 + net(k))
      residual  — 输出: grey → grey + MLP([u,k,x0])

    三个方向正交, 任意组合 (包括全关 = 标准 GNN)。

    Args:
        input_dim:  特征维度, 默认 5。
        hidden_dim: LC 隐藏节点数, 默认 40。
        deep_lb:    启用深层特征编码器。
        deep_la:    启用时变衰减率。
        residual:   启用显式残差校正。
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 40,
        deep_lb: bool = False,
        deep_la: bool = False,
        residual: bool = False,
    ):
        # 先调基类构造, 再覆盖/追加深度组件
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self._deep_lb = deep_lb
        self._deep_la = deep_la
        self._residual = residual

        # ── 深层 LB ──────────────────────────────────────────
        if deep_lb:
            self.LB = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(),
                nn.Linear(16, 8),  nn.ReLU(),
                nn.Linear(8, 1),
            )

        # ── 时变 LA ──────────────────────────────────────────
        if deep_la:
            self.LA_net = nn.Sequential(
                nn.Linear(4, 4), nn.ReLU(),
                nn.Linear(4, 1),
            )

        # ── 残差校正 ─────────────────────────────────────────
        if residual:
            self.res_net = nn.Sequential(
                nn.Linear(input_dim + 2, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 1),
            )

    def _has_deep(self) -> bool:
        return self._deep_lb or self._deep_la or self._residual

    # ── 前向传播 ──────────────────────────────────────────────

    def forward(
        self, k: torch.Tensor, u: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        # 指数衰减核: 常数 → 时变 a(k)
        if self._deep_la:
            k_enc = torch.stack([
                torch.sin(k), torch.cos(k),
                torch.sin(2 * k), torch.cos(2 * k),
            ], dim=-1)
            delta_a = self.LA_net(k_enc).squeeze(-1)
            a = F.softplus(-self.w11 + delta_a)    # a(k) > 0
        else:
            a = -self.w11

        decay = torch.exp(-a * k)                  # e^(−a·k)

        # 特征编码: 5 个运行参数 → 驱动项
        driving = self.LB(u).squeeze(-1)

        # 灰组合 (公式 4)
        grey = (x0 - driving) * decay + driving

        # 残差校正: 灰趋势 + 深度网络输出
        if self._residual:
            k_u = k.unsqueeze(-1)
            x0_b = x0.expand_as(k_u)
            res_in = torch.cat([u, k_u, x0_b], dim=-1)
            grey = grey + self.res_net(res_in).squeeze(-1)

        # Sigmoid 隐藏层 → 线性输出层
        h = torch.sigmoid(self.LC(grey.unsqueeze(-1)))
        return self.LD(h).squeeze(-1)

    # ── 灰系数提取 ──────────────────────────────────────────────

    def extract_grey_coeffs(self) -> "tuple[float, np.ndarray]":
        w11 = float(self.w11.item())
        a = -w11
        if self._deep_lb:
            first_linear = self.LB[0]
            lb_w = first_linear.weight.data[0, :5].cpu().numpy()
        else:
            lb_w = self.LB.weight.data[0].cpu().numpy()
        b = lb_w / w11
        return a, b

    # ── 预测 ────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        if self._has_deep():
            return _predict_with_network(self, data)
        else:
            return super().predict(data)


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _predict_with_network(model: GNN, data: dict) -> dict:
    """用训练好的 forward() 做全模型逐点推断。"""
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(data["X_all"], dtype=torch.float32)
        k_all = torch.arange(X_all.shape[0], dtype=torch.float32)
        x0 = torch.tensor(data["y_all_ago"][0], dtype=torch.float32)

        y_ago = torch.zeros(X_all.shape[0], dtype=torch.float64)
        y_ago[0] = x0.double()
        for i in range(1, X_all.shape[0]):
            y_ago[i] = model(k_all[i:i+1], X_all[i:i+1], x0).double()

    y_pred_norm = (y_ago[1:] - y_ago[:-1]).numpy()
    y_pred = (
        y_pred_norm * (data["target_dmax"] - data["target_dmin"])
        + data["target_dmin"]
    )

    a, bs = model.extract_grey_coeffs()
    return {
        "y_pred": y_pred,
        "y_true": data["y_all_raw"][1:],
        "a": a,
        "bs": bs,
    }
