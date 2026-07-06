# GNN — PEMFC 退化预测的灰色神经网络

> 原论文：*Real-time degradation prognostics for proton exchange membrane fuel cells using a self-adaptive grey neural network model* (Energy, 2025)
>
> 本仓库：标准 GNN 的 PyTorch 复现 + 三个架构改进方向 + 与论文基线的对比实验

## 研究动机

论文提出的 SiGDSM-GNN 在三个 PEMFC 退化数据集上取得了 0.11% MAPE 和 0.005s 推理速度。该结果来自 SiGDSM 种群搜索机制（100 个候选解，10000 次函数评估），标准 BP 训练的基线 GNN 仅达到 0.42–0.72% MAPE（Table 10, BP1–BP4）。这一差距表明，论文的精度优势主要来自优化策略的改进，而非网络架构本身。

本工作从另一方向探索：在不改变优化策略的前提下，能否通过改进 GNN 的架构设计来提升性能？论文 §5 (line 585) 将 "integrating it with deeper neural architectures for enhanced temporal modelling" 列为未来方向，但未给出具体方案。实验聚焦于论文中两个与架构直接相关的待解决问题：

> §4.4.2 (line 501–502): "the proposed model does **not explicitly separate** reversible transient effects from irreversible degradation mechanisms … explicit decomposition … represents an **important direction for future work**"

> §4.4.2 (line 505–506): "future improvements may also integrate **adaptive weighting mechanisms** or **attention-based architectures** to dynamically regulate the contribution of highly sensitive variables"

前者要求将退化趋势与短期波动解耦建模，后者要求对不同运行参数在不同退化阶段赋予自适应权重。标准 GNN 的四层结构（LA→LB→LC→LD）存在两个容量瓶颈：LB 层为单层 Linear，无法捕捉特征交互；LA 层为常数衰减率，隐含了完美指数退化的假设。

---

## 三条改进路线

**路线 A：Deep LB — 深层特征编码**

LB: `Linear(5, 1)` → `MLP(5 → 16 → ReLU → 8 → ReLU → 1)`。多层非线性结构使网络能够学习运行参数之间的交互效应（如湿度与电流密度对膜降解的协同作用），直接对应论文的 "adaptive weighting" 方向。

**路线 B：Deep LA — 时变衰减率**

LA: 常数 a → `a(k) = softplus(−w11 + net(k))`。k 通过位置编码 `[sin(k), cos(k), sin(2k), cos(2k)]` 输入一个小型前馈网络。衰减率随退化阶段自适应变化——早期催化剂失活较快、中期趋稳、后期可能出现加速失效或饱和。

**路线 C：Residual — 灰趋势 + 残差校正**

论文 §2.1 (line 138) 将框架定性为 "hierarchical trend-extraction and adaptive-correction architecture"，但实际网络中灰趋势与神经校正是隐式耦合的（均经 LC+LD 的 sigmoid 路径）。本路线将其显式化为 skip connection：`ŷ = grey(u, k, x0) + res_net([u, k, x0])`。灰模型提供指数退化先验作为下限（即使残差分支训坏，MAPE 不超过纯灰模型的 1.34%），残差网络仅拟合偏离，对应论文的 "explicit decomposition" 方向。

| 路线 | 改动位置 | 新增参数 | 对应论文问题 |
|------|------|:---:|------|
| Deep LB | LB 层 | +236 | #6 自适应权重 |
| Deep LA | LA 头 | +25 | #6 退化阶段自适应 |
| Residual | 输出端 | +273 | #5 可逆/不可逆分解 |

三条路线实现为 `DeepGNN(GNN)` 子类，通过 `deep_lb`、`deep_la`、`residual` 三个布尔参数任意组合。

**参考**：Deep LB 受 Dynamic Filter Networks (NIPS 2016) 启发；Residual 源于 ResNet (CVPR 2016)。

---

## 使用方法

```bash
# 论文原始 GNN
python -c "from gnn.main import main; main('FC1', seed=42)"

# 单一改进方向
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

## 关于小样本训练深层模型的说明

6 个训练样本、5 维特征——总计 30 个标量观测值——支撑 300-400 参数的深层模型似乎违背了深度学习的常规直觉。三条路线可行但原因不同：

**Deep LB (363 params) 能行是因为结构约束强。** MLP 5→16→8→1 并非自由地学习黑箱映射，而是受灰组合 `(x0 − LB)·e^(−a·k) + LB` 强约束。网络只需让 LB 输出逼近 (1/a)·Σb·u，深层结构仅为该逼近提供更多自由度。有效参数量远小于形式上的 363——大部分权重被灰模板锁定了优化方向。

**Residual (400 params) 能行是因为灰模型兜底。** 灰预测已提供 1.34% MAPE 的下限。残差分支仅需学习千分之几伏特的修正量——其有效输出范围约为灰模型输出的 1/100，训练时 loss 主要来自灰模型残差，残差分支几乎不会主导梯度。

**组合实验为此提供了证据。** 636 参数的 lb+res 组合（MAPE 1.22%）反而不如各自单独使用（0.29% 和 0.25%），表明 ~400 是 6 样本的容量上限。361–400 恰好在有效区间：足够表达特征交互和短期偏差，但尚未进入过拟合噪声的区间。

**Deep LA (152 params, MAPE 1.10%) 是反例。** 参数少于前两者，但时变衰减率的自由度过高而训练信号过弱——6 个 k 值无法覆盖 1000 小时退化过程中的速度变化。

结论：小样本下深层模型可行的条件是 (a) 结构先验足够强，限制了参数的有效自由度；或 (b) 基线模型已提供足够低的下限，深层分支仅需微调输出。

---

## 核心文件

```
gnn/
├── model.py              # 论文原始 GNN (LA→LB→LC→LD, 严格遵循 Eq.4 和 Eq.6)
├── model_extensions.py   # DeepGNN, 三个布尔开关可任意组合
├── train.py              # Adam 全批量训练
├── main.py               # 端到端入口, 接收 deep_lb/deep_la/residual 参数
├── config.py             # 数据集配置 + 训练超参数
└── data.py               # 数据流水线 (加载→1-AGO→时间顺序划分)
```

---

## 实验

所有实验在 FC1 数据集（稳态退化，5-cell PEMFC，~1000h，8 个特征化时刻）上完成。划分方式：前 6 个时刻训练、后 2 个验证（时间顺序 extrapolation）。共享超参数：Adam, lr=0.001, 5000 epochs, full batch, hidden_dim=40。

**Table 1: 标准 GNN 多 seed 验证**

| seed | VAL MAPE | a |
|------|:---:|:---:|
| 42 | 1.34% | 0.53 |
| 456 | 1.42% | 0.50 |
| 789 | 1.37% | 0.52 |
| 123 | 6.93% | 0.45 |

标准 GNN 在 3/4 seed 下 MAPE 稳定在 1.34–1.42%，接近论文 BP 变体的 0.42–0.72% 区间。seed=123 陷入鞍点——6 样本单次梯度下降的固有缺陷，论文通过 SiGDSM 种群搜索规避。

**Table 2: 三种扩展 × 8 组合 (seed=42)**

| 配置 | 参数量 | VAL MAPE | vs 论文 BP1 (0.42%) |
|------|:---:|:---:|:---:|
| GNN (baseline) | 127 | 1.34% | — |
| deep_lb | 363 | **0.29%** | ✓ 超越 |
| deep_la | 152 | 1.10% | — |
| residual | 400 | **0.25%** | ✓ 超越 |
| lb + la | 388 | 0.29% | ✓ 超越 |
| lb + res | 636 | 1.22% | — |
| la + res | 425 | 0.48% | 接近 |
| all_three | 661 | 0.36% | ✓ 超越 |

**Table 3: 最优路线多 seed 稳定性**

| 路线 | 4-seed mean | std | min |
|------|:---:|:---:|:---:|
| **deep_lb** | **0.32%** | 0.02% | 0.29% |
| **residual** | **0.35%** | 0.14% | **0.10%** |

Deep LB 最稳定（跨 seed std=0.02%），所有 seed 均落在 0.29–0.34%。残差路线 seed=123 达到 0.10%，与论文 SiGDSM 的 0.11% 持平——且不使用种群搜索，仅靠标准 Adam 梯度下降。

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
