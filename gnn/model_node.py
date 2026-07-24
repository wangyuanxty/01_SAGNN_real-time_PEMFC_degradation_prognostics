"""Neural ODE 版 GNN — 用连续时间微分方程替代离散灰组合。

核心思想 (Chen et al. 2018):
  离散层堆叠 → 连续向量场 dx/dt = f(t, x)
  灰微分方程提供物理先验, 神经网络校正残差。

架构:
  ODE 函数:  f(t, x) = −a·x + b·u(t) + net(t, x)
  求解器:    RK4 固定步长, 从 t=0 到 t=k 分段积分
  训练:      MSE(预测AGO, 真实AGO), 梯度通过求解器回传
  预测:      solve ODE → IAGO → 逆归一化

优势:
  - 天然处理不等间隔时间点 (论文的 35-185h 间隔)
  - 可在任意未来时刻预测, 不受限于 k=0..7
  - 灰微分方程 + 残差校正 = 物理先验 + 数据拟合

参考:
  - Chen et al., Neural Ordinary Differential Equations, NeurIPS 2018
  - https://arxiv.org/abs/1806.07366
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# RK4 ODE 求解器
# ═══════════════════════════════════════════════════════════════════

def _rk4_step(
    f: callable, t: torch.Tensor, x: torch.Tensor,
    dt: float, u: torch.Tensor,
) -> torch.Tensor:
    """单步 RK4。

    Args:
        f: (t, x, u) -> dx/dt, 返回与 x 同形的张量。
        t: scalar, 当前时间。
        x: (*,) 当前状态。
        dt: 步长。
        u: (*,) 当前时刻的特征 (在步内视为常数)。
    Returns:
        x 在 t+dt 处的值。
    """
    k1 = f(t, x, u)
    k2 = f(t + dt / 2, x + dt * k1 / 2, u)
    k3 = f(t + dt / 2, x + dt * k2 / 2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _integrate(
    f: callable,
    x0: torch.Tensor,
    t_start: float,
    t_end: float,
    dt: float,
    u: torch.Tensor,
) -> torch.Tensor:
    """从 t_start 到 t_end 用 RK4 积分, 区间内 u 视为常数。

    Args:
        f: ODE 函数 (t, x, u) -> dx/dt。
        x0: 初始状态, scalar。
        t_start, t_end: 积分起止时间。
        dt: 步长。
        u: 该区间内的特征向量 (5,)。
    Returns:
        t_end 时刻的状态, scalar。
    """
    n_steps = max(1, int((t_end - t_start) / dt))
    actual_dt = (t_end - t_start) / n_steps
    t = torch.tensor(t_start, dtype=torch.float32)
    x = x0
    for _ in range(n_steps):
        x = _rk4_step(f, t, x, float(actual_dt), u)
        t = t + actual_dt
    return x


# ═══════════════════════════════════════════════════════════════════
# Neural ODE GNN
# ═══════════════════════════════════════════════════════════════════

class NeuralODEGNN(nn.Module):
    """Neural ODE 版灰色神经网络。

    ODE 函数: dx/dt = −a·x + Σb_i·u_i + net(t, x)
    灰项 (−a·x + Σb·u) 对应论文公式 (3) 的白化微分方程,
    net(t, x) 为神经网络残差校正。

    Args:
        input_dim:   特征维度, 默认 5。
        hidden_dim:  net 隐藏节点数, 默认 16。
        dt:          RK4 积分步长, 默认 0.1。
        use_net:     是否启用神经网络校正 (False = 纯灰 ODE)。
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 16,
        dt: float = 0.1,
        use_net: bool = True,
    ):
        super().__init__()
        self.dt = dt
        self.input_dim = input_dim
        self.use_net = use_net

        # ── 灰模型参数 ──────────────────────────────────────────
        # 指数衰减核: w11 → a = −w11 > 0, 保证指数衰减
        self.w11 = nn.Parameter(torch.tensor(-0.2))

        # 特征编码层: 5 个运行参数 → 驱动系数, 用于 Σb·u 项
        self.LB = nn.Linear(input_dim, 1, bias=False)

        # ── 神经网络残差校正 ────────────────────────────────────
        if use_net:
            # 输入: [t, x] — 时间和当前状态
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.LB.weight, -0.25, 0.25)
        if self.use_net:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.25, 0.25)
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    # ── ODE 函数 ──────────────────────────────────────────────

    def _ode_func(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """右端函数 f(t, x) = −a·x + Σb·u + net(t, x)。

        Args:
            t: scalar。
            x: scalar, 当前 AGO 状态。
            u: (5,) 特征向量。
        Returns:
            dx/dt, scalar。
        """
        a = -self.w11                             # 退化速度 a > 0
        grey_drift = -a * x                       # 灰模型指数衰减项: −a·x
        grey_input = self.LB(u.unsqueeze(0)).squeeze(-1).squeeze(0)  # 特征驱动项: Σb·u

        if self.use_net:
            t_in = t.unsqueeze(0).unsqueeze(0)    # (1, 1)
            x_in = x.detach().unsqueeze(0).unsqueeze(0)  # (1, 1)
            inp = torch.cat([t_in, x_in], dim=-1)  # (1, 2)
            correction = self.net(inp).squeeze()    # scalar
        else:
            correction = 0.0

        return grey_drift + grey_input + correction

    # ── 前向传播 ──────────────────────────────────────────────

    def forward(
        self, k: torch.Tensor, u: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """分段积分: 从 t=0 到每个 k_i, u 分段常数。

        Args:
            k:  (batch,) 目标时间步 [k_0, k_1, ...]。
            u:  (batch, 5) 各段特征。
            x0: scalar, AGO 初始条件。
        Returns:
            (batch,) 各 k_i 处的 AGO 预测值。
        """
        x_current = x0
        t_current = 0.0
        results = []

        for i in range(len(k)):
            t_target = float(k[i].item())
            if t_target > t_current:
                x_current = _integrate(
                    self._ode_func, x_current,
                    t_current, t_target, self.dt,
                    u[i],
                )
            results.append(x_current)
            t_current = t_target

        return torch.stack(results)

    # ── 灰系数提取 ──────────────────────────────────────────────

    def extract_grey_coeffs(self) -> "tuple[float, np.ndarray]":
        w11 = float(self.w11.item())
        a = -w11
        lb_w = self.LB.weight.data[0].cpu().numpy()
        b = lb_w / w11
        return a, b

    # ── 预测 ────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        """ODE 求解全轨迹 → IAGO → 逆归一化。"""
        self.eval()
        with torch.no_grad():
            X_all = torch.tensor(data["X_all"], dtype=torch.float32)
            k_all = torch.arange(X_all.shape[0], dtype=torch.float32)
            x0 = torch.tensor(data["y_all_ago"][0], dtype=torch.float32)

            y_ago = self(k_all, X_all, x0).double()

        y_pred_norm = (y_ago[1:] - y_ago[:-1]).numpy()
        y_pred = (
            y_pred_norm * (data["target_dmax"] - data["target_dmin"])
            + data["target_dmin"]
        )

        a, bs = self.extract_grey_coeffs()
        return {
            "y_pred": y_pred,
            "y_true": data["y_all_raw"][1:],
            "a": a,
            "bs": bs,
        }
