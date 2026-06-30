# GNN 实现设计（标准灰色神经网络）

> 2026-06-30 | 状态: 已批准  
> 实现: Sapnken et al. "Real-time degradation prognostics for PEMFC using a self-adaptive grey neural network model" — 标准 GNN 部分（§2.1–2.2）

## 范围

用 PyTorch 实现论文的**标准 4 层灰色神经网络**（LA→LB→LC→LD），包含数据加载、归一化/AGO/IAGO 预处理、训练循环和评估。SiGDSM 自智能优化机制（§2.3）**不在本次范围内**。采用论文基线 GNN 的标准反向传播训练。

## 模块结构

```
gnn/
├── __init__.py
├── config.py          # DatasetConfig, TrainConfig, 模块级单例常量
├── data.py            # 数据流水线: 加载 → 提取特征化点 → 归一化 → AGO → 划分
├── model.py           # GNN 模型类 + 灰系数提取 + 时间响应预测
├── train.py           # 训练循环（适配小样本）
└── main.py            # 端到端入口: 配置 → 数据 → 训练 → 评估 → 绘图
```

## 数据流水线 (`data.py`)

### 数据来源

- **FC1**: 稳态退化, 3 个 Excel 文件, 8 个特征化时刻 = [0, 48, 185, 348, 515, 658, 823, 991] h
- **FC2**: 动态加载 + 5 kHz 纹波, 1 Excel + 1 分号分隔 CSV, 8 个特征化时刻 = [0, 35, 182, 343, 515, 666, 830, 1016] h

### 特征列（5 个）

| 列名 | 物理含义 |
|------|---------|
| `Utot (V)` | 电堆总电压（自回归输入） |
| `HrAIRFC (%)` | 进气空气相对湿度 |
| `PinH2 (mbara)` | 氢气入口压力 |
| `TinWAT (°C)` | 冷却水入口温度 |
| `I (A)` | 负载电流 |

### 目标列

`Utot (V)` — 与特征 1 同列, 自回归结构。模型学到的是: **给定时刻 t 的全部 5 个运行测量值, 预测该时刻对应的 AGO 编码电压退化水平。**

### 流水线步骤

1. **load_raw_data**: 拼接数据集的所有 part 文件。Excel 用 `pd.read_excel`, CSV 用 `pd.read_csv(sep=";", decimal=",")`, 注意欧式数字格式。
2. **extract_char_points**: 对每个特征化时刻 t, 取 `abs(df["Time (h)"] - t)` 最小的行。
3. **build_samples**: 选取 5 个特征列 → `X: (8, 5)`, 目标列 → `y_raw: (8,)`。
4. **normalize**: Min-max 缩放到 [0, 1]。X 按列独立归一化（`dmin`, `dmax` shape 为 `(5,)`）, y 作为标量归一化。保留 `dmin`、`dmax` 用于逆变换。
5. **AGO**（论文公式 2）: `y_ago = cumsum(y_norm)`, 仅对归一化后的目标做累加, 输出 shape `(8,)`。
6. **划分**: 6 训练 / 2 验证, 固定种子随机 shuffle。同时保留 `X_all`、`y_all_ago`、`y_all_raw` 用于全序列评估和画图。

### 返回字典

```python
{
    "X_train": (6, 5), "X_val": (2, 5),
    "y_train_ago": (6,), "y_val_ago": (2,),
    "y_train_raw": (6,), "y_val_raw": (2,),
    "X_all": (8, 5), "y_all_ago": (8,), "y_all_raw": (8,),
    "X_dmin": (5,), "X_dmax": (5,),
    "target_dmin": float, "target_dmax": float,
    "char_times": list[float], "name": str,
}
```

## 模型架构 (`model.py`)

### 各层定义

| 层 | PyTorch 模块 | 输入维度 | 输出维度 | 激活函数 | 角色 |
|----|-------------|---------|---------|---------|------|
| LA | `nn.Identity` | 5 | 5 | 恒等 | 接收归一化特征，直通传递 |
| LB | `nn.Linear(5, 3, bias=True)` | 5 | 3 | 恒等 | 灰化层；第一个神经元的权重编码 a, b_i |
| LC | `nn.Linear(3, 10)` + sigmoid | 3 | 10 | Sigmoid | 隐藏非线性层 |
| LD | `nn.Linear(10, 1)` | 10 | 1 | 恒等 | 输出层，输出 AGO 空间的电压预测值 |

**LB 输出维度为何取 3**: 第一个 LB 神经元通过 5 个输入权重 + 1 个偏置共 6 个参数来编码 6 个灰系数（1 个 a + 5 个 b_i）。剩余 2 个神经元在进入 sigmoid 隐藏层之前提供额外的特征变换能力。维度 3 是在论文无明确规定下的最小合理选择。

### 灰系数提取（论文公式 6）

LB 为 `nn.Linear(5, 3, bias=True)`。训练完成后, 从 LB 的第一个神经元提取:

```python
w = model.LB.weight[0]   # shape (5,)
b = model.LB.bias[0]     # scalar
w11 = w[0]

a  = -w11                         # 发展系数
b1 = w[1] / w11                   # 电压影响系数
b2 = w[2] / w11                   # 湿度影响系数
b3 = w[3] / w11                   # 氢压影响系数
b4 = w[4] / w11                   # 温度影响系数
b5 = b  / w11                     # 电流影响系数（由 bias 编码）
```

5 个权重 + 1 个 bias = 6 个灰参数, 恰好对应 1 个 a + 5 个 b_i。

### 训练时的前向传播

```python
def forward(self, x):
    # x: (batch, 5) — 各特征化时刻的归一化特征
    x = self.LA(x)                          # 恒等, (batch, 5)
    x = self.LB(x)                          # Linear, (batch, 3)
    x = torch.sigmoid(self.LC(x))           # Linear + Sigmoid, (batch, 10)
    x = self.LD(x)                          # Linear, (batch, 1)
    return x.squeeze(-1)                    # (batch,) — AGO 空间预测值
```

### 预测（训练后走解析路径, 不走 forward）

训练完成后, 预测**不走** `forward()`。而是提取灰系数, 代入时间响应函数做解析预测:

```python
def predict(self, data):
    a, bs = self.extract_grey_coeffs()       # a: float, bs: (5,) ndarray
    x0 = data["X_all"][0, 0]                 # 第一个点的归一化电压值 x^(0)(1)
    k_vals = np.arange(len(data["char_times"]))   # [0, 1, ..., 7]

    y_ago_pred = np.zeros(8)
    y_ago_pred[0] = x0

    for i in range(1, 8):
        u = data["X_all"][i]                 # 当前点的 5 个特征
        u_sum = np.dot(bs, u)                # Σ b_i * u_i
        # 时间响应函数（公式 4）
        y_ago_pred[i] = (x0 - u_sum / a) * np.exp(-a * k_vals[i]) + u_sum / a

    # IAGO 逆累加（公式 8）
    y_pred_norm = np.diff(y_ago_pred)       # shape (7,)
    # 逆归一化
    y_pred = y_pred_norm * (data["target_dmax"] - data["target_dmin"]) + data["target_dmin"]

    return {"y_pred": y_pred, "y_true": data["y_all_raw"][1:], "a": a, "bs": bs}
```

**关键设计决策**: 预测时使用每个特征化时刻的实际 5 个特征值（含电压）。这对应论文的评估方式——给定已知运行条件, 评估模型学到的退化映射是否准确。IAGO 后第一点（k=0）无差分值, 因此 `y_pred` 和 `y_true` 均为 7 个点。

### 权重初始化

所有 `nn.Linear` 层: `nn.init.uniform_(w, 0.0, 0.5)`, bias 同样。照论文 §3.3。

## 训练 (`train.py`)

### 超参数（照论文 §3.3）

| 参数 | 值 |
|------|-----|
| LC 隐藏节点数 | 10 |
| LB 输出维度 | 3 |
| LC 激活函数 | Sigmoid |
| LD 激活函数 | Linear（恒等） |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 权重初始化范围 | U(0, 0.5) |
| 最大迭代次数 | 1000 |
| 损失函数 | MSELoss |
| 训练/验证划分 | 6/2 |

### 训练循环

```python
def train(model, data, config):
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = MSELoss()
    X = torch.tensor(data["X_train"], dtype=torch.float32)
    y = torch.tensor(data["y_train_ago"], dtype=torch.float32)

    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.6f}")

    return model, losses
```

### 小样本适配说明

- 仅 6 个训练样本 → 全批量梯度下降（不做 mini-batch）
- 使用 Adam, 小样本下自适应学习率更稳定
- 此样本量下, 几百个 epoch 内训练 loss 接近零是正常现象
- 提供 `run(seed)` 包装函数, 多次独立运行取均值以获得可靠统计

## 评估 (`main.py`)

### 评估指标

- **MAPE (%)** = mean(|(y_true - y_pred) / y_true|) × 100
- **MSE** = mean((y_true - y_pred)²)
- **RMSE** = sqrt(MSE)

### 输出

1. 控制台: 最终训练 loss、验证集 MAPE/MSE/RMSE、提取的灰系数 a 和 b1..b5
2. 图 1: 电压退化曲线——预测值 vs 真实值, 覆盖全部 8 个特征化时刻
3. 图 2: 训练 loss 曲线

### 入口函数

```python
def main(dataset_name="FC1", seed=42):
    cfg = FC1_CONFIG if dataset_name == "FC1" else FC2_CONFIG
    train_cfg = TRAIN_CONFIG
    data = load_dataset(cfg, seed=seed)
    model = GNN()
    model, losses = train(model, data, train_cfg)
    results = evaluate(model, data)
    print_metrics(results)
    plot_degradation(results, data)
    plot_loss(losses, train_cfg)
```

## 错误处理

- **文件不存在**: 抛出 `FileNotFoundError`, 打印解析后的完整路径
- **特征化点数不匹配**: 抛出 `ValueError`, 打印期望 vs 实际
- **归一化分母为零**（某列全为常数）: 分母替换为 1.0
- **训练后 a ≈ 0**（灰模型奇点）: 打印警告, 该 seed 下模型不可用

## 测试策略

1. **单元: normalize 回路** — `inverse_normalize(normalize(x)) ≈ x`, 浮点容差内
2. **单元: AGO/IAGO 回路** — `iago_sequence(ago_sequence(y)) ≈ y[1:]`, 浮点容差内
3. **单元: 灰系数提取** — 构造已知 LB 权重的 `GNN`, 验证 `extract_grey_coeffs()` 返回正确的 a 和 b_i
4. **集成: 数据流水线形状** — `load_dataset(FC1_CONFIG)` 返回 `X_train(6,5)`, `y_train_ago(6,)`, `X_val(2,5)`, `y_val_ago(2,)`
5. **集成: FC1 完整流水线** — 训练 + 评估, 验证 MAPE < 2%（标准 GNN 基线; 论文 PSO-GNN 约 1.3%）
