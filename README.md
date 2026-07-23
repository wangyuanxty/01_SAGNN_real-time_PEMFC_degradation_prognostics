# 基于深度化 GNN 的 PEMFC 退化预测开放问题研究

> 原论文：Sapnken et al., *Real-time degradation prognostics for PEMFC using a self-adaptive grey neural network model*, Energy, 2025
>
> 本仓库复现论文的标准 GNN，针对论文提出的五个开放问题分别给出架构层面的解决方案，并通过对比实验验证有效性。

## 标准 GNN

论文 §2.1–2.2 提出的标准 GNN 将灰微分方程的时间响应函数嵌入四层神经网络。复现严格遵循公式 (4) 和公式 (6)。6 个训练样本、2 个验证样本（时间顺序划分）、Adam 梯度下降。127 参数，3/4 seeds 稳定在 MAPE 1.34–1.42%。

---

## 问题 1: 评估偏差 — 固定划分 → 时间顺序划分

论文 §3.1 (line 289)：

> "a simple fixed train-validation split **may introduce evaluation bias** due to the particular choice of validation points"

论文用 70-30 随机划分（6 训练/2 验证），这意味着模型可能用 T=823h 的数据训练、去验证 T=48h 的数据——对退化预测而言没有意义。论文自身引入 LOOCV 纠正此问题，但 Table 4 全文最醒目的对比表仍使用固定划分。

**修改。** 将划分策略改为时间顺序：前 6 个特征化时刻（0–658h）训练，后 2 个（823h、991h）验证。模型必须从过去预测未来，而非从全部时刻随机采样。这一改动使得所有指标都在严格的 extrapolation 条件下报告。

---

## 问题 2: 可逆退化与不可逆退化未分离

论文 §4.1 (line 320) + §4.4.2 (line 501–502)：

> "the proposed model does **not explicitly separate** reversible transient effects from irreversible degradation mechanisms … explicit decomposition … represents an **important direction for future work**"

**问题。** PEMFC 运行中会出现可逆电压恢复——停机重启后电压短暂回升、极化曲线测试后膜再水化。标准 GNN 靠 1-AGO 的平滑效果把这些波动当噪声压掉，但 "mitigates" 不是 "eliminates"。如果不显式分离，一次大幅电压回升会让模型误判退化趋势。

**路线：Residual — 灰趋势 + 残差分解。**

论文 §2.1 (line 138) 将框架定性为 "hierarchical trend-extraction and adaptive-correction architecture"，但实际网络中灰趋势与神经校正是隐式耦合的。将其显式化：

```
grey = (x0 − LB(u))·e^(−a·k) + LB(u)      # 灰模型: 不可逆退化骨架
res  = res_net([u, k, x0])                  # 残差网络: 可逆波动 + 其他偏差
ŷ    = grey + res
```

灰模型提供指数退化先验。残差网络只拟合偏离——有效输出量级约为灰模型的 1/100。即使残差分支训坏，退化为纯灰模型（MAPE 1.34%）。

**实现。** `DeepGNN(residual=True)`。`res_net` 为 MLP(7→16→ReLU→8→ReLU→1)，273 额外参数。

**结果。** 4 seeds，MAPE 均值 0.35% ± 0.14%，min 0.10%。seed=123 达到 0.10%——与论文 SiGDSM 的 0.11% 持平，无需种群搜索。

参考：ResNet (CVPR 2016), https://arxiv.org/abs/1512.03385

---

## 问题 3: 对高度敏感变量缺乏自适应加权

论文 §4.4.2 (line 505–506)：

> "future improvements may also integrate **adaptive weighting mechanisms** or **attention-based architectures** to dynamically regulate the contribution of highly sensitive variables"

**问题。** 论文 §4.4.2 的灵敏度分析显示负载电流 ±20% 波动导致 APE 从正常水平飙升至 1.5%。标准 GNN 的 LB 是 `nn.Linear(5, 1)`——单层线性隐含各特征独立贡献的假设。但 PEMFC 退化存在特征交互——高湿度加速膜降解仅在高电流密度下显著，低湿度+高温的组合效应不是加和关系。

**路线：Deep LB — 多层特征编码。**

```
u(5) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → lb_out
```

三层非线性使网络能学习二阶以上特征交互。"湿度高+电流大"与"湿度低+电流大"的退化速率可以被区分。灰系数仍通过第一层权重提取，可解释性保留。

**实现。** `DeepGNN(deep_lb=True)`。236 额外参数。

**结果。** 4 seeds，MAPE 均值 0.32% ± 0.02%，所有组合中最稳定。

---

## 问题 4: 更深层的时序建模架构

论文 §5 (line 585) 将以下方向明确列为 future work：

> "integrating it with **deeper neural architectures** for enhanced temporal modelling"

论文只提了方向，没有给具体方案。标准 GNN 的时序建模仅靠一个常数衰减率 a——退化速度在整个生命周期内固定。

**路线。** Deep LB 和 Residual 分别从特征编码和残差校正两个维度加深网络。此外，Deep LA（时变衰减率 `a(k) = softplus(−w11 + net(k))`）尝试让退化速度随阶段自适应变化。Deep LA 当前 MAPE 1.10%（劣于 baseline），因为 6 个 k 值不足以训练有意义的时变函数，但方法本身为未来在更长时序数据上的应用保留了可能性。

---

## 问题 5: 不等间隔时间点

论文未将此列为显式开放问题，但标准 GM(1,N) 理论要求等间隔数据，而论文的 8 个特征化时刻间隔为 35–185h 不等。时间响应函数 `e^(−a·k)` 使用序数索引 k=0..7，隐含 Δk=1 的等步长假设。

**路线：Neural ODE — 连续时间微分方程。**

```
dx/dt = −a·x + b·u + net(t, x)
```

用 RK4 求解器在连续时间轴上积分，u(t) 在各区间内分段常数。时间以实际步长（而非序数）进入积分器。纯灰 ODE（6 参数）等价于解析解，MAPE 0.49%。加神经网络校正后（343 参数）MAPE 0.52%。连续时间形式天然支持未来任意时刻预测。

参考：Chen et al., Neural ODEs, NeurIPS 2018, https://arxiv.org/abs/1806.07366

---

## 汇总

| # | 论文开放问题 | 出处 | 我们的方案 | 路线 | MAPE |
|---|------|------|------|------|:---:|
| 1 | 固定划分有评估偏差 | §3.1 | 改为时间顺序划分 | 数据协议 | 1.34% |
| 2 | 可逆/不可逆退化未分离 | §4.1, §4.4.2 | 灰趋势+残差分解 | Residual | **0.35%** |
| 3 | 缺乏自适应特征权重 | §4.4.2 | 多层特征编码 | Deep LB | **0.32%** |
| 4 | 更深时序建模架构 | §5 | 三条深度化路线 | 全部 | — |
| 5 | 不等间隔时间点 | GM(1,N) 隐含 | 连续时间微分方程 | Neural ODE | 0.52% |

---

## 深层模型在小样本上的可行性

6 个训练样本、363–400 参数——违反常规深度学习直觉。两组机制保证可行：

- **灰模板约束。** `(x0 − LB)·e^(−a·k) + LB` 将 LB 的输出空间强约束为一维标量，有效自由度远小于形式参数量。
- **灰模型兜底。** Residual 模式下，灰分支提供 1.34% MAPE 下限，残差分支仅需微调。

证据：636 参数的 lb+res 组合（MAPE 1.22%）劣于各自单独使用（0.29%、0.25%），表明 ~400 是 6 样本容量上限。

---

## 使用方法

```bash
# 标准 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 问题 2 — 可逆/不可逆分离
python -c "from gnn.main import main; main('FC1', seed=42, residual=True)"

# 问题 3 — 自适应权重
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True)"

# 问题 5 — 不等间隔
python -c "from gnn.main import main; main('FC1', seed=42, node=True)"

# 任意组合
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True, residual=True)"
```

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN
├── model_extensions.py   # DeepGNN (deep_lb / deep_la / residual)
├── model_node.py         # Neural ODE GNN (RK4 求解器)
├── train.py / main.py    # 训练与评估
├── config.py             # 数据集配置 + 超参数
└── data.py               # 数据流水线
```

---

## 实验

FC1（稳态退化, 8 个特征化时刻, 前 6 训练后 2 验证）。Adam, lr=0.001, 5000 epochs, full batch。

**标准 GNN 各 seed：**

| seed | 42 | 456 | 789 | 123 |
|------|:---:|:---:|:---:|:---:|
| MAPE | 1.34% | 1.42% | 1.37% | 6.93% |

seed=123 陷入鞍点——6 样本梯度下降的固有问题。

**各路线对比 (seed=42)：**

| 路线 | 参数 | MAPE | 对应问题 |
|------|:---:|:---:|------|
| GNN baseline | 127 | 1.34% | — |
| Residual | 400 | **0.25%** | 可逆/不可逆分离 |
| Deep LB | 363 | **0.29%** | 自适应权重 |
| Neural ODE | 343 | 0.52% | 不等间隔 |
| Deep LA | 152 | 1.10% | deeper architectures |
| lb+res | 636 | 1.22% | (过参数化) |

**最优路线多 seed：**

| 路线 | mean | std | min |
|------|:---:|:---:|:---:|
| Deep LB | **0.32%** | 0.02% | 0.29% |
| Residual | **0.35%** | 0.14% | **0.10%** |

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
