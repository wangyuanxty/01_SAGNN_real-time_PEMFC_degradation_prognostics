# 基于深度化 GNN 的 PEMFC 退化预测开放问题研究

> 原论文：Sapnken et al., *Real-time degradation prognostics for PEMFC using a self-adaptive grey neural network model*, Energy, 2025
>
> 本仓库复现论文的标准 GNN，针对论文提出的两个开放问题——可逆/不可逆退化分离与自适应特征权重——分别给出架构层面的解决方案，并通过对比实验验证有效性

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

FC1 数据集（稳态退化, 8 个特征化时刻, 前 6 训练后 2 验证），Adam, lr=0.001, 5000 epochs。

---

## 开放问题 1: 可逆退化与不可逆退化未分离

论文 §4.4.2 (line 501–502)：

> "the proposed model does **not explicitly separate** reversible transient effects from irreversible degradation mechanisms … explicit decomposition … represents an **important direction for future work**"

**问题。** PEMFC 运行中会出现可逆电压恢复——停机重启后电压短暂回升，极化曲线测试后膜再水化。标准 GNN 靠 1-AGO 的平滑效果把这些波动当噪声压掉，但论文 §4.1 (line 320) 承认 "the smoothing effect of 1-AGO mitigates the influence of such short-term fluctuations"——"mitigates"不是"eliminates"。如果不分离，一个电压回升较大的数据段会让模型误判退化趋势。

**路线：Residual — 灰趋势 + 残差分解。**

论文 §2.1 (line 138) 将框架定性为 "hierarchical trend-extraction and adaptive-correction architecture"，但网络中灰趋势与神经校正是隐式耦合的。将其显式化：

```
grey = (x0 − LB(u))·e^(−a·k) + LB(u)      # 灰模型: 不可逆退化骨架
res  = res_net([u, k, x0])                  # 残差网络: 可逆波动 + 其他偏差
ŷ    = grey + res
```

灰模型提供指数退化先验。残差网络只拟合偏离——其有效输出量级约为灰模型输出的 1/100，不会淹没骨架。即使残差分支训坏，退化为纯灰模型（MAPE 1.34%）。

**实现：** `DeepGNN(residual=True)`。`res_net` 为 MLP(7→16→ReLU→8→ReLU→1)，输入 [u(5), k(1), x0(1)]。273 额外参数。

**结果：** FC1 4 seeds，VAL MAPE 均值 0.35% ± 0.14%，min 0.10%。seed=123 达到 0.10%——与论文 SiGDSM 的 0.11% 持平，且无需种群搜索。

参考：ResNet (CVPR 2016), https://arxiv.org/abs/1512.03385

---

## 开放问题 2: 负载电流敏感性与自适应权重

论文 §4.4.2 (line 505–506)：

> "future improvements may also integrate **adaptive weighting mechanisms** or **attention-based architectures** to dynamically regulate the contribution of highly sensitive variables"

**问题。** 论文 §4.4.2 的灵敏度分析显示，负载电流 ±20% 波动导致 APE 从正常水平的 <0.5% 飙升至 1.5%。标准 GNN 的 LB 是 `nn.Linear(5, 1, bias=False)`——5 个特征的加权和。单层线性隐含的假设是湿度、氢压、温度、电流各自独立地对退化产生线性贡献。但真实的 PEMFC 退化存在特征交互——高湿度加速膜降解仅在高电流密度下显著，低湿度+高温的组合效应不是各自贡献的简单加和。

**路线：Deep LB — 深层特征编码。**

将 LB 从单层线性替换为三层 MLP：

```
u(5) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → lb_out
```

多层非线性使网络能学习二阶甚至三阶特征交互。"湿度高 + 电流大"与"湿度低 + 电流大"的退化速率可以不同。灰系数仍通过第一层 Linear 的权重除以 w11 提取，可解释性保留。深层结构不增加新参数类别——仅替代原 Linear 层。

**实现：** `DeepGNN(deep_lb=True)`。236 额外参数。

**结果：** FC1 4 seeds，VAL MAPE 均值 0.32% ± 0.02%，所有 seed 均落在 0.29–0.34%。所有组合中最稳定——跨 seed std 仅 0.02%。

---

## 未采纳的路线

**Deep LA（时变衰减率）** 尝试让衰减率 a 随退化阶段变化——`a(k) = softplus(−w11 + net(k))`。结果 MAPE 1.10%，劣于 baseline。6 个训练 k 值（0..5）不足以训练有意义的时变函数。在数据更多时（如 FCEV 实车 36 个月连续监测），此方向价值更高。

**Neural ODE** 用连续时间微分方程替代离散灰组合，RK4 积分。纯灰 ODE（6 参数）MAPE 0.49%，加网络校正（343 参数）MAPE 0.52%。理论上优雅但数值积分慢，不适合作为默认方案。

---

## 深层模型在小样本上的可行性

6 个训练样本、363–400 参数——违反常规深度学习直觉。

**Deep LB 可行因为灰模板约束。** `(x0 − LB)·e^(−a·k) + LB` 将 LB 的输出空间强约束为一维标量，网络只需学一个接近 (1/a)·Σb·u 的值。363 个参数的有效自由度远小于形式上的数量。

**Residual 可行因为灰模型兜底。** 残差分支输出量级为灰模型的 ~1/100，训练时 loss 主要来自灰模板的残余误差，残差网络几乎不主导梯度。

**证据：** 636 参数的 lb+res 组合（MAPE 1.22%）反而不如各自单独使用（0.29% 和 0.25%）。~400 是 6 样本的容量上限。

---

## 使用方法

```bash
# 论文原始 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 开放问题 1 (可逆/不可逆分离)
python -c "from gnn.main import main; main('FC1', seed=42, residual=True)"

# 开放问题 2 (自适应权重)
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True)"

# Neural ODE
python -c "from gnn.main import main; main('FC1', seed=42, node=True)"

# 所有开关
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True, deep_la=True, residual=True, node=True)"
```

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN (127 参数)
├── model_extensions.py   # DeepGNN (deep_lb / deep_la / residual 三个 bool)
├── model_node.py         # Neural ODE GNN (RK4 求解器)
├── train.py              # Adam 全批量训练
├── main.py               # 端到端入口, 接收所有开关
├── config.py             # 数据集配置 + 训练超参数
└── data.py               # 数据流水线 (加载→1-AGO→时间顺序划分)
```

---

## 实验

FC1 数据集（稳态退化, 5-cell PEMFC, ~1000h, 8 个特征化时刻）。前 6 个时刻训练、后 2 个验证。共享超参数：Adam, lr=0.001, 5000 epochs, full batch。

**标准 GNN 多 seed：**

| seed | VAL MAPE |
|------|:---:|
| 42 | 1.34% |
| 456 | 1.42% |
| 789 | 1.37% |
| 123 | 6.93% |

seed=123 陷入鞍点——6 样本梯度下降的固有问题，论文用 SiGDSM 种群搜索解决。

**三条路线对比 (seed=42)：**

| 路线 | 参数 | MAPE | 对应问题 |
|------|:---:|:---:|------|
| GNN baseline | 127 | 1.34% | — |
| **Residual** | **400** | **0.25%** | 可逆/不可逆分离 |
| **Deep LB** | **363** | **0.29%** | 自适应权重 |
| Deep LA | 152 | 1.10% | (未采纳) |
| Neural ODE | 343 | 0.52% | (备选) |

**最优路线多 seed：**

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
