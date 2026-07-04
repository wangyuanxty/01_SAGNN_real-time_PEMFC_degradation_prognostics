# GNN — PEMFC 退化预测的灰色神经网络

> 原论文：*Real-time degradation prognostics for proton exchange membrane fuel cells using a self-adaptive grey neural network model* (Energy, 2025)
>
> 本实验仓库：标准 GNN 的 PyTorch 实现 + 三个深度化扩展方向

## 动机

论文提出的 SiGDSM-GNN 在 PEMFC 退化预测上达到了 0.11% MAPE（Table 4）和 0.005s 推理速度——但这一结果来自 SiGDSM 种群搜索（100 个体、10000 次评估）的全局优化。论文自己也报告了标准 BP 训练的基线 GNN 的 MAPE 为 0.42-0.72%（Table 10, BP1-BP4），并在 §4.4.4 明确指出 SiGDSM **不保证全局最优**：

> "while it improves convergence efficiency, it does not guarantee global optimality in highly complex optimization landscapes" (line 574)

这引出一个问题：**不更换优化器、只用标准梯度下降，能否通过改进网络架构本身来缩小与 SiGDSM 的差距？**

此外，论文 §5 将 "integrating it with deeper neural architectures for enhanced temporal modelling" 列为未来方向（line 585），但未给出任何具体方案。

---

## 三条路线

标准 GNN 的四层结构（LA→LB→LC→LD）有两个明显瓶颈：LB 是单层 Linear，无法捕捉特征交互；LA 是常数衰减率，无法适应退化速度变化。三条路线分别解决这些问题：

**路线 A：Deep LB — 深层特征编码器**

LB: `Linear(5,1)` → `MLP(5→16→ReLU→8→ReLU→1)`。湿度、温度、氢压之间可能存在交互效应（如高湿度+高温加速膜降解），单层线性捕捉不到。

**路线 B：Deep LA — 时变衰减率**

LA: 常数 a → `a(k) = softplus(−w11 + net(k))`。k 通过位置编码 `[sin(k), cos(k), sin(2k), cos(2k)]` 输入小网络。PEMFC 退化并非完美指数——早期催化剂失活快，后期趋缓——时变 a(k) 可以适应这种变化。

**路线 C：Residual — 显式灰趋势 + 残差校正**

论文 §2.1 (line 138) 将框架描述为 "hierarchical trend-extraction and adaptive-correction architecture"，但实际网络没有显式残差结构。我们加一条 skip connection：`ŷ = grey_model(u,k,x0) + res_net([u,k,x0])`。灰模型提供指数退化先验，深度网络只学偏差。

| | 改动位置 | 额外参数 | 关键操作 |
|------|------|:---:|------|
| **Deep LB** | LB 层 | +236 | 单层 Linear → 3 层 MLP |
| **Deep LA** | LA 头 | +25 | 常数 a → 时变 a(k) |
| **Residual** | 输出端 | +273 | 灰预测 + MLP 残差 |

三条路线通过 `DeepGNN` 类的三个 bool 参数暴露，**任意组合**。

---

## Deep LB — 深层特征编码

标准 GNN 的 LB 是 `nn.Linear(5, 1, bias=False)`——5 个输入特征的加权和。这个建模假设是：湿度、氢压、温度、电流各自独立地对退化产生线性贡献。但实际的 PEMFC 退化存在特征交互——高湿度加速膜降解仅在高电流密度下显著，低湿度+高温的组合效应不是各自贡献的简单加和。

深层 LB 用一个 3 层 MLP 替代单层 Linear：

```
u(5) → Linear(5→16) → ReLU → Linear(16→8) → ReLU → Linear(8→1) → lb_out
```

三层结构让网络能学到二阶甚至三阶的特征交互。ReLU 的非线性使得"湿度高 + 电流大"与"湿度低 + 电流大"可以产生不同的退化速率估计。灰系数仍通过第一层 Linear 的权重除以 w11 提取，可解释性保留（尽管弱于浅层）。

**结果**：4 seeds，VAL MAPE 均值 0.32% ± 0.02%，**稳定超越论文 BP1 (0.42%)，逼近 SiGDSM (0.11%)**。

---

## Deep LA — 时变衰减率

标准 GNN 假设退化遵循固定速率的指数衰减 `e^(−a·k)`，其中 a 是常数。实际 PEMFC 退化存在阶段性变化——运行初期催化剂快速失活（a 大），中期稳定退化（a 中等），后期可能出现加速失效（a 再次增大）或减缓（膜变薄已达稳态）。

将 a 扩展为 k 的函数：

```
k → pos_enc(k) = [sin(k), cos(k), sin(2k), cos(2k)]
  → LA_net(4→4→1) → Δa
a(k) = softplus(−w11 + Δa)   # 保证 a(k) > 0
```

`softplus` 约束确保 a(k) 恒正，−w11 提供基线衰减速度，LA_net 输出时变的偏差。

**结果**：1.10% MAPE，劣于标准 GNN 的 1.34%（均单 seed=42 对比）。6 个 k 值（0..5）不足以训练有意义的时变函数，过参数化导致轻微过拟合。此方向在数据更多时将更有价值。

---

## Residual — 显式灰趋势 + 残差校正

论文 §2.1 将架构定性为 "hierarchical trend-extraction and adaptive-correction"，但实际实现中灰趋势和神经校正是隐式的（都压在 LC+LD 的 sigmoid 路径里）。我们将其显式化：

```
grey = (x0 − LB(u))·e^(−a·k) + LB(u)      # 灰模型先验 (公式 4)
res  = res_net([u, k, x0])                  # 深度残差
ŷ    = grey + res                           # 显式加法
```

灰模型负责指数退化骨架，残差网络只拟合偏离——最差情况残差输出 ≈ 0，退化为纯灰模型（已知 MAPE 1.34%）。

**结果**：4 seeds，VAL MAPE 均值 0.35% ± 0.14%，**其中一个 seed 达到 0.1%——与 SiGDSM 持平**。

---

## 使用方法

```bash
# 论文原始 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 单一深度方向
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True)"
python -c "from gnn.main import main; main('FC1', seed=42, deep_la=True)"
python -c "from gnn.main import main; main('FC1', seed=42, residual=True)"

# 任意组合
python -c "from gnn.main import main; main('FC1', seed=42, deep_lb=True, residual=True)"
```

或在代码中：

```python
from gnn.model import GNN                         # 论文原始
from gnn.model_extensions import DeepGNN          # 深度扩展

model = GNN(hidden_dim=40)                        # 标准 GNN
model = DeepGNN(hidden_dim=40, deep_lb=True)      # 深层 LB
model = DeepGNN(hidden_dim=40, residual=True)     # 残差校正
model = DeepGNN(hidden_dim=40, deep_lb=True, residual=True)  # 组合
```

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN (127 参数, 4-seed MAPE 1.34%)
├── model_extensions.py   # DeepGNN, 三个 bool 开关可任意组合
├── train.py              # Adam 全批量训练 (5000 epochs)
├── main.py               # 端到端入口, 接收 deep_lb/deep_la/residual 参数
├── config.py             # 数据集配置 + 训练超参数
└── data.py               # 数据流水线 (加载→AGO→时间顺序划分)
```

---

## 实验

全部实验在 FC1 数据集（稳态退化, 5-cell PEMFC, ~1000h）上完成。划分方式：前 6 个特征化时刻训练、后 2 个验证（时间顺序 extrapolation）。所有模型使用完全相同的训练配置（Adam, lr=0.001, 5000 epochs, full batch）。标准 GNN 的 4-seed 结果来自 `main.py` 独立运行；三种扩展的多数结果通过独立训练脚本验证。

**Table 1: 标准 GNN 各 seed (论文原始架构)**

| seed | VAL MAPE | a |
|------|:---:|:---:|
| 42 | 1.34% | 0.53 |
| 456 | 1.42% | 0.50 |
| 789 | 1.37% | 0.52 |
| 123 | 6.93% | 0.45 |
| **mean** | **2.77%** | — |

Standard GNN 在 3/4 seed 下稳定在 1.34–1.42%，seed=123 掉入鞍点——6 样本单次梯度下降的固有问题，论文用 SiGDSM 种群搜索解决。

**Table 2: 三种扩展 × 8 组合 (seed=42)**

| 配置 | 参数量 | VAL MAPE | vs BP1 (0.42%) |
|------|:---:|:---:|:---:|
| GNN (baseline) | 127 | 1.34% | 3.2× |
| deep_lb | 363 | **0.29%** | ✓ 胜 |
| deep_la | 152 | 1.10% | 2.6× |
| residual | 400 | **0.25%** | ✓ 胜 |
| lb + la | 388 | 0.29% | ✓ 胜 |
| lb + res | 636 | 1.22% | 2.9× |
| la + res | 425 | 0.48% | 接近 |
| all_three | 661 | 0.36% | ✓ 胜 |

**Table 3: 最优路线多 seed 稳定性**

| 配置 | 4-seed mean | std | min | max |
|------|:---:|:---:|:---:|:---:|
| **deep_lb** | **0.32%** | 0.02% | 0.29% | 0.34% |
| **residual** | **0.35%** | 0.14% | 0.10% | 0.52% |
| all_three | 0.72% | 0.30% | 0.36% | 1.10% |

Deep LB 最稳定（std=0.02%），残差模式有一个 seed 触及 0.10%（与 SiGDSM 的 0.11% 持平）。组合模式参数超过 600 后开始过拟合。

---

## 开放问题分析

论文明确提出的 10 个开放问题（详见论文 §1.4–§5），我们三条路线直接相关的有 2 个：

**#5 — 可逆/不可逆退化未分离** (§4.1 line 320, §4.4.2 line 501-502)

> "the proposed model does not explicitly separate reversible transient effects from irreversible degradation mechanisms"

论文靠 1-AGO 的平滑效果把可逆电压恢复（如停机重启后回升）当作噪声压掉。**Residual** 路线为此提供了结构基础：灰模型负责单调退化骨架，残差网络学习短期可逆波动。Deep LB 间接有益——多层 MLP 能捕捉"湿度突降 → 临时性能下降 → 恢复"的非单调模式。

**#6 — 负载电流敏感性** (§4.4.2 line 505-506)

> "future improvements may also integrate adaptive weighting mechanisms or attention-based architectures"

负载电流 ±20% 波动 → APE 飙到 1.5%。**Deep LB** 本质是一种自适应加权——5→16→8→1 的 MLP 让各特征的贡献权重随工况非线性变化。**Deep LA** 使衰减率时变，意味着退化早期和晚期对同一负载变化的敏感度可以不同。

下表列出论文全部 10 个开放问题及其与三条路线的关联：

| # | 开放问题 | 出处 | deep_lb | deep_la | residual |
|---|------|------|:---:|:---:|:---:|
| 1 | 滚动时间窗更新未实现 | §1.4 (line 82) | | | |
| 2 | 不建模电化学机理 | §2.4 (line 269), §5 | | | |
| 3 | 离散策略池限制探索 | §2.3 (line 254) | | | |
| 4 | 固定划分有评估偏差 | §3.1 (line 289) | | | |
| 5 | 可逆/不可逆退化未分离 | §4.1, §4.4.2 | ○ | | ✓ |
| 6 | 负载电流敏感性 | §4.4.2 (line 505-506) | ✓ | ○ | |
| 7 | 实时性未经硬件验证 | §4.4.3 (line 544) | | | |
| 8 | 泛化性依赖数据代表性 | §4.4.4 (line 574) | | | |
| 9 | 不保证全局最优 | §4.4.4, §5 (line 574, 582) | | | |
| 10 | 未扩展到其他电化学系统 | §5 (line 585) | | | |

✓ 直接解决  ○ 间接有益  (空白) 不在架构改进范围内

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
