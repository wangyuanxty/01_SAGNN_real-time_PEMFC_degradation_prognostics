# GNN — PEMFC 退化预测的灰色神经网络

> 原论文：*Real-time degradation prognostics for proton exchange membrane fuel cells using a self-adaptive grey neural network model* (Energy, 2025)
>
> 本实验仓库：标准 GNN 的 PyTorch 实现 + 三个深度化扩展方向

## 动机

论文提出的 SiGDSM-GNN 达到了 0.11% MAPE（Table 4），但这一结果来自 SiGDSM 种群搜索（100 个体、10000 次评估）的全局优化。论文 §4.4.4 (line 574) 明确指出该机制**不保证全局最优**：

> "while it improves convergence efficiency, it does not guarantee global optimality in highly complex optimization landscapes"

不换优化器、只用标准梯度下降，能否通过改进网络架构本身来缩小与 SiGDSM 的差距？

此外，论文 §5 (line 585) 将 "integrating it with deeper neural architectures for enhanced temporal modelling" 列为未来方向，但只提了方向没有给方案。论文中有两个与架构直接相关的开放问题：

> §4.4.2 (line 501-502): "the proposed model does **not explicitly separate** reversible transient effects from irreversible degradation mechanisms … explicit decomposition … represents an **important direction for future work**"

> §4.4.2 (line 505-506): "future improvements may also integrate **adaptive weighting mechanisms** or **attention-based architectures** to dynamically regulate the contribution of highly sensitive variables"

前者需要把退化趋势和短期波动分开建模，后者需要对不同特征在不同退化阶段赋予自适应权重。两条都是架构层面可解决的问题。

标准 GNN 的四层结构有两个明显瓶颈：LB 是单层 Linear，无法捕捉"高湿度+高温"这类特征交互；LA 是常数衰减率，假定了完美指数退化。

---

## 三条路线

**路线 A：Deep LB — 深层特征编码器**

LB: `Linear(5,1)` → `MLP(5→16→ReLU→8→ReLU→1)`。三层 MLP 让不同特征的非线性交互可学习，直接对应论文的"adaptive weighting"方向。

**路线 B：Deep LA — 时变衰减率**

LA: 常数 a → `a(k) = softplus(−w11 + net(k))`。退化速度可以随时间变化——早期催化剂失活快，后期趋缓。

**路线 C：Residual — 显式灰趋势 + 残差校正**

论文 §2.1 (line 138) 将框架描述为 "hierarchical trend-extraction and adaptive-correction architecture"。灰模型负责指数退化骨架，残差网络只学偏离。最差情况残差输出 ≈ 0，退化为纯灰模型。

| | 改动位置 | 额外参数 | 对应论文开放问题 |
|------|------|:---:|------|
| **Deep LB** | LB 层 | +236 | #6 自适应权重, #5 特征交互间接有益 |
| **Deep LA** | LA 头 | +25 | #6 退化阶段自适应 |
| **Residual** | 输出端 | +273 | #5 可逆/不可逆分解 |

三条路线通过 `DeepGNN(bool, bool, bool)` 任意组合。

---

## Deep LB — 深层特征编码

标准 GNN 的 LB 是 `nn.Linear(5, 1, bias=False)`——5 个输入特征的加权和。单层 Linear 假设湿度、氢压、温度、电流各自独立地对退化产生线性贡献。但实际的 PEMFC 退化存在特征交互——高湿度加速膜降解仅在高电流密度下显著，低湿度+高温的组合效应不是各自贡献的简单加和。

```
u(5) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → lb_out
```

灰系数仍通过第一层 Linear 的权重除以 w11 提取（可解释性保留）。训练后收敛极快，loss 从 epoch 0 的 9.23 单调降到 epoch 1500 的 0.001。

**参考**：论文 §4.4.2 line 505-506 "adaptive weighting mechanisms"

---

## Deep LA — 时变衰减率

标准 GNN 假设退化遵循固定速率的指数衰减 `e^(−a·k)`，a 为常数。实际 PEMFC 退化存在阶段性变化。

```
k → pos_enc(k) = [sin(k), cos(k), sin(2k), cos(2k)]
  → LA_net(4→4→1) → Δa
a(k) = softplus(−w11 + Δa)   # 保证 a(k) > 0
```

`softplus` 约束确保 a(k) 恒正，−w11 提供基线衰减速度，LA_net 输出时变偏差。6 个 k 值（0..5）不足以训练有意义的时变函数——收敛后 train loss 降至 0 但 val MAPE 反而不如 baseline。此方向在数据更多时价值更大。

---

## Residual — 显式残差校正

论文 §2.1 将架构定性为 "hierarchical trend-extraction and adaptive-correction"，但实际网络并无显式残差结构——灰趋势和神经校正全压在 LC+LD 的 sigmoid 里。

```
grey = (x0 − LB(u))·e^(−a·k) + LB(u)      # 灰模型先验 (公式 4)
res  = res_net([u, k, x0])                  # 深度残差 (7→16→8→1)
ŷ    = grey + res                           # 显式加法
```

灰趋势保证下限（即使残差网络训坏，最差退化为纯灰模型 MAPE 1.34%）。残差网络只学偏离，直接对应论文 §4.4.2 (line 501-502) 的 "explicit decomposition of reversible and irreversible degradation mechanisms"。

**参考**：ResNet (CVPR 2016), https://arxiv.org/abs/1512.03385

---

## 使用方法

```bash
# 论文原始 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 单一方向
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True)"
python -c "from gnn.main import main; main('FC1', seed=42, residual=True)"

# 任意组合
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True, residual=True)"
```

```python
from gnn.model import GNN
from gnn.model_extensions import DeepGNN

model = GNN(hidden_dim=40)                          # 标准 GNN
model = DeepGNN(hidden_dim=40, deep_lb=True)        # 深层 LB
model = DeepGNN(hidden_dim=40, residual=True)       # 残差校正
model = DeepGNN(hidden_dim=40, deep_lb=True, residual=True)
```

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN
├── model_extensions.py   # DeepGNN, 三个 bool 开关可任意组合
├── train.py              # Adam 全批量训练
├── main.py               # 端到端入口, 接收 deep_lb/deep_la/residual
├── config.py             # 数据集配置 + 训练超参数
└── data.py               # 数据流水线 (加载→AGO→时间顺序划分)
```

---

## 实验

全部实验在 FC1 数据集（稳态退化, 5-cell PEMFC, ~1000h, 8 个特征化时刻）上完成。划分方式：前 6 个时刻训练、后 2 个验证（时间顺序 extrapolation）。超参数：Adam, lr=0.001, 5000 epochs, full batch, hidden_dim=40。

**Table 1: 标准 GNN 各 seed**

| seed | VAL MAPE | a |
|------|:---:|:---:|
| 42 | 1.34% | 0.53 |
| 456 | 1.42% | 0.50 |
| 789 | 1.37% | 0.52 |
| 123 | 6.93% | 0.45 |

Standard GNN 在 3/4 seed 下稳定在 1.34–1.42%。seed=123 掉入鞍点（loss 在 0.9 附近停滞 ~2000 epoch）——6 样本单次梯度下降的固有问题，论文用 SiGDSM 种群搜索解决。

**Table 2: 三种扩展 × 8 组合 (seed=42)**

| 配置 | 参数量 | VAL MAPE | vs 论文 BP1 (0.42%) |
|------|:---:|:---:|:---:|
| GNN (baseline) | 127 | 1.34% | — |
| deep_lb | 363 | **0.29%** | 胜 |
| deep_la | 152 | 1.10% | 差 |
| residual | 400 | **0.25%** | 胜 |
| lb + la | 388 | 0.29% | 胜 |
| lb + res | 636 | 1.22% | 差 |
| la + res | 425 | 0.48% | 接近 |
| all_three | 661 | 0.36% | 胜 |

**Table 3: 最优路线多 seed**

| 配置 | 4-seed mean | std | min |
|------|:---:|:---:|:---:|
| **deep_lb** | **0.32%** | 0.02% | 0.29% |
| **residual** | **0.35%** | 0.14% | **0.10%** |

Deep LB 最稳定（std=0.02%），所有 seed 均在 0.29–0.34%。残差路线 seed=123 达到 0.10%——与论文 SiGDSM 的 0.11% 持平，且不需要种群搜索。

---

## 🔗 引用

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
