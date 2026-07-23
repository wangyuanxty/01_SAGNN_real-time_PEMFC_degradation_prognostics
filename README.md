# 基于深度化 GNN 的 PEMFC 退化预测开放问题研究

> 原论文：Sapnken et al., *Real-time degradation prognostics for PEMFC using a self-adaptive grey neural network model*, Energy, 2025
>
> 本仓库复现论文的标准 GNN，针对论文提出的两个与架构直接相关的开放问题——可逆/不可逆退化分离与自适应特征权重——分别给出解决方案，并通过对比实验验证有效性

## 标准 GNN

论文 §2.1–2.2 提出的标准 GNN 将灰微分方程的时间响应函数嵌入四层神经网络。复现采用 PyTorch，严格遵循公式 (4) 的灰组合和公式 (6) 的权重-系数映射关系：

```python
class GNN(nn.Module):
    def forward(self, k, u, x0):
        a = -self.w11                              # a > 0, 指数衰减速度
        la = torch.exp(-a * k)                     # e^(−a·k)
        lb = self.LB(u).squeeze(-1)                # 驱动项, 训练时不除 w11
        grey = (x0 - lb) * la + lb                 # 公式 (4)
        h = torch.sigmoid(self.LC(grey.unsqueeze(-1)))
        return self.LD(h).squeeze(-1)
```

FC1 数据集（稳态退化, 8 个特征化时刻, 前 6 训练后 2 验证），Adam, lr=0.001, 5000 epochs。标准 GNN 在 3/4 seed 下 MAPE 稳定在 1.34–1.42%，接近论文 BP 变体的 0.42–0.72% 区间。seed=123 陷入鞍点——6 样本梯度下降的固有问题，论文用 SiGDSM 种群搜索解决。

论文共提出 10 个开放问题。其中两个落在架构改进范围内：

> §4.4.2 (line 501–502): "the proposed model does **not explicitly separate** reversible transient effects from irreversible degradation mechanisms … explicit decomposition … represents an **important direction for future work**"

> §4.4.2 (line 505–506): "future improvements may also integrate **adaptive weighting mechanisms** or **attention-based architectures** to dynamically regulate the contribution of highly sensitive variables"

前者要求将退化趋势与短期波动解耦建模，后者要求对不同运行参数赋予自适应权重。标准 GNN 的四层结构有两个容量瓶颈：LB 为单层 Linear，隐含特征独立线性贡献的假设；网络缺乏显式的趋势-波动分解结构。

其余 8 个问题（滚动时间窗部署、电化学机理建模、优化理论、硬件验证、泛化性等）落在优化策略、部署工程或物理建模层面，不在本工作的架构改进范围内。

---

## 开放问题 1: 可逆退化与不可逆退化未分离

论文 §4.4.2 (line 501–502)。

**问题。** PEMFC 运行中会出现可逆电压恢复——停机重启后电压短暂回升，极化曲线测试后膜再水化。标准 GNN 靠 1-AGO 的平滑效果把这些波动当噪声压掉（论文 §4.1 line 320: "the smoothing effect of 1-AGO mitigates the influence of such short-term fluctuations"），但非显式分解。一个电压回升较大的数据段足以让模型误判退化趋势。

**路线：Residual — 灰趋势 + 残差分解。**

论文 §2.1 将框架定性为 "hierarchical trend-extraction and adaptive-correction architecture"，但网络中灰趋势与神经校正是隐式耦合的。将其显式化：

```
grey = (x0 − LB(u))·e^(−a·k) + LB(u)      # 灰模型: 不可逆退化骨架
res  = res_net([u, k, x0])                  # 残差网络: 可逆波动 + 其他偏差
ŷ    = grey + res
```

灰模型提供指数退化先验。残差网络只拟合偏离——其有效输出量级约为灰模型的 1/100，不会淹没骨架。最差情况残差分支训坏，退化为纯灰模型（MAPE 1.34%）。

Deep LB 间接有益于该问题：多层特征编码能学到"湿度突降 + 温度不变 → 膜干涸 → 电压临时下降后恢复"这类非单调交互模式，单层 Linear 不可能。

**实现：** `DeepGNN(residual=True)`。`res_net` 为 MLP(7→16→ReLU→8→ReLU→1)，输入 [u(5), k(1), x0(1)]。273 额外参数。

**结果：** FC1 4 seeds，VAL MAPE 均值 0.35% ± 0.14%，min 0.10%。seed=123 的 0.10% 与论文 SiGDSM 的 0.11% 持平，且无需种群搜索。

参考：ResNet (CVPR 2016), https://arxiv.org/abs/1512.03385

---

## 开放问题 2: 负载电流敏感性与自适应权重

论文 §4.4.2 (line 505–506)。

**问题。** 论文 §4.4.2 的灵敏度分析显示，负载电流 ±20% 波动导致 APE 从正常水平的 <0.5% 飙升至 1.5%。标准 GNN 的 LB 是 `nn.Linear(5, 1, bias=False)`——5 个特征的加权和。单层线性隐含的假设是各运行参数对退化独立地产生线性贡献。但实际 PEMFC 退化存在特征交互——高湿度加速膜降解仅在高电流密度下显著，低湿度+高温的组合效应不是各自贡献的简单加和。

**路线：Deep LB — 深层特征编码。**

将 LB 从单层线性替换为三层 MLP：

```
u(5) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → lb_out
```

多层非线性使网络能学习二阶甚至三阶特征交互。"湿度高 + 电流大"与"湿度低 + 电流大"的退化速率可以不同，直接对应论文的 "adaptive weighting"。灰系数仍通过第一层 Linear 的权重除以 w11 提取，可解释性保留。

Deep LA 也有助于此：时变衰减率 a(k) 意味着退化早期和晚期对同一负载变化的敏感度可以不同。

**实现：** `DeepGNN(deep_lb=True)`。236 额外参数。

**结果：** FC1 4 seeds，VAL MAPE 均值 0.32% ± 0.02%，所有 seed 均落在 0.29–0.34%。三个路线中最稳定——跨 seed std 仅 0.02%。

---

## 额外探索

**Deep LA（时变衰减率）。** 将常数 a 扩展为 `a(k) = softplus(−w11 + net(k))`。动机是论文 §5 (line 585) 的 "deeper neural architectures for enhanced temporal modelling"，但未作为正式开放问题。结果 MAPE 1.10%，劣于 baseline——6 个训练 k 值不足以训练有意义的时变函数，确认了该方向在数据量更大时才有实际价值。

**Neural ODE。** 用连续时间微分方程 `dx/dt = −a·x + Σb·u + net(t,x)` 替代离散灰组合，RK4 积分。天然处理不等间隔时间点（论文特征化时刻间隔 35–185h）。纯灰 ODE（6 参数）MAPE 0.49%，加网络校正（343 参数）MAPE 0.52%。理论上优雅但数值积分慢，作为备选方案保留在 `model_node.py`。

参考：Chen et al., Neural ODE, NeurIPS 2018, https://arxiv.org/abs/1806.07366

---

## 深层模型在小样本上的可行性

6 个训练样本、363–400 参数。

**Deep LB 可行因为灰模板约束。** `(x0 − LB)·e^(−a·k) + LB` 将 LB 的输出空间强约束为一维标量。363 个参数的有效自由度远小于形式上的数量。

**Residual 可行因为灰模型兜底。** 残差分支输出量级为灰模型的 ~1/100，训练时几乎不主导梯度。

**Deep LA 不可行因为信号太弱。** 时变函数需要足够的 k 密度才能学到有意义的时序变化——6 个点不够。

**证据：** 636 参数的 lb+res 组合（MAPE 1.22%）反而不如各自单独使用（0.29% 和 0.25%），确认 ~400 是 6 样本的容量上限。

---

## 使用方法

```bash
# 论文原始 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 开放问题 1 (可逆/不可逆分离)
python -c "from gnn.main import main; main('FC1', seed=42, residual=True)"

# 开放问题 2 (自适应权重)
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True)"

# 额外探索
python -c "from gnn.main import main; main('FC1', seed=42, deep_la=True)"
python -c "from gnn.main import main; main('FC1', seed=42, node=True)"

# 任意组合
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True, residual=True)"
```

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN (127 参数)
├── model_extensions.py   # DeepGNN (deep_lb / deep_la / residual)
├── model_node.py         # Neural ODE GNN (RK4 求解器)
├── train.py              # Adam 全批量训练
├── main.py               # 端到端入口
├── config.py             # 数据集配置 + 训练超参数
└── data.py               # 数据流水线 (加载→1-AGO→时间顺序划分)
```

---

## 实验

FC1 数据集（稳态退化, 5-cell PEMFC, ~1000h, 8 个特征化时刻）。前 6 个时刻训练、后 2 个验证。共享超参数：Adam, lr=0.001, 5000 epochs, full batch。

| 路线 | 参数 | VAL MAPE | 对应 |
|------|:---:|:---:|------|
| GNN baseline | 127 | 1.34% | — |
| **Residual** | **400** | **0.25%** | 问题 1: 可逆/不可逆分离 |
| **Deep LB** | **363** | **0.29%** | 问题 2: 自适应权重 |
| Deep LA | 152 | 1.10% | 额外探索 |
| Neural ODE | 343 | 0.52% | 额外探索 |

**最优路线多 seed 稳定性：**

| 路线 | mean | std | min |
|------|:---:|:---:|:---:|
| **Deep LB** | **0.32%** | 0.02% | 0.29% |
| **Residual** | **0.35%** | 0.14% | **0.10%** |

---

## 引用

```bibtex
@article{sapnken2025real,
  title={Real-time degradation prognostics for proton exchange membrane fuel cells
         using a self-adaptive grey neural network model},
  author={Sapnken, Flavian Emmanuel and Wang, Yong and Posso, Fausto and
          Bangoup Ntegmi, Ghislain Junior and Molu, Reagan Jean Jacques and
          Xie, Naiming},
  journal={Energy},
  year={2025}
}
```
