# GNN Implementation Design (Standard Grey Neural Network)

> 2026-06-30 | Status: approved
> Implements: Sapnken et al. "Real-time degradation prognostics for PEMFC using a self-adaptive grey neural network model" — standard GNN portion (§2.1–2.2)

## Scope

Implement the **standard 4-layer Grey Neural Network** (LA→LB→LC→LD) using PyTorch, including data loading, normalization/AGO/IAGO preprocessing, training loop, and evaluation. The SiGDSM optimization mechanism (§2.3) is **out of scope**. The model is trained with standard backpropagation per the paper's baseline GNN.

## Module Structure

```
gnn/
├── __init__.py
├── config.py          # DatasetConfig, TrainConfig, module-level singletons
├── data.py            # Data pipeline: load → extract → normalize → AGO → split
├── model.py           # GNN model class + grey coefficient extraction + time response
├── train.py           # Training loop with small-sample adaptations
└── main.py            # End-to-end entry: config → data → train → eval → plot
```

## Data Pipeline (`data.py`)

### Input sources

- **FC1**: steady-state degradation, 3 Excel parts, 8 characterization times at [0, 48, 185, 348, 515, 658, 823, 991] h
- **FC2**: dynamic loading with 5 kHz ripples, 1 Excel + 1 semicolon-separated CSV, 8 characterization times at [0, 35, 182, 343, 515, 666, 830, 1016] h

### Feature columns (5)

| Column | Physical meaning |
|--------|-----------------|
| `Utot (V)` | Stack total voltage (autoregressive input) |
| `HrAIRFC (%)` | Air relative humidity |
| `PinH2 (mbara)` | Hydrogen inlet pressure |
| `TinWAT (°C)` | Cooling water inlet temperature |
| `I (A)` | Load current |

### Target column

`Utot (V)` — same column as feature 1. The model learns: given all 5 operating measurements at characterization time t, predict the corresponding AGO-encoded voltage degradation level.

### Pipeline steps

1. **load_raw_data**: Read and concatenate all part files for a dataset. Excel files via `pd.read_excel`, CSV files via `pd.read_csv(sep=";", decimal=",")`.
2. **extract_char_points**: For each characterization time t, select the row where `abs(df["Time (h)"] - t)` is minimized.
3. **build_samples**: Select the 5 feature columns → `X: (8, 5)`, and the target column → `y_raw: (8,)`.
4. **normalize**: Min-max scaling to [0, 1], independently per column for X (`dmin`, `dmax` shape `(5,)`), and as a scalar for y. Preserve normalization parameters for inverse transform.
5. **AGO** (Eq. 2): `y_ago = cumsum(y_norm)` applied to the normalized target only. Output shape `(8,)`.
6. **Split**: 6 train / 2 val, randomly shuffled with fixed seed. The data dict also retains `X_all`, `y_all_ago`, `y_all_raw` for full-sequence evaluation and plotting.

### Return value

A dict: `{X_train(6,5), X_val(2,5), y_train_ago(6,), y_val_ago(2,), y_train_raw(6,), y_val_raw(2,), X_all(8,5), y_all_ago(8,), y_all_raw(8,), X_dmin(5,), X_dmax(5,), target_dmin(float), target_dmax(float), char_times, name}`

## Model Architecture (`model.py`)

### Layer definitions

| Layer | PyTorch module | Input | Output | Activation | Role |
|-------|---------------|-------|--------|------------|------|
| LA | `nn.Identity` | 5 | 5 | Identity | Receives normalized features |
| LB | `nn.Linear(5, 3, bias=True)` | 5 | 3 | Identity | Grey layer; first neuron weights encode a, b_i |
| LC | `nn.Linear(3, 10)` + sigmoid | 3 | 10 | Sigmoid | Hidden nonlinear layer |
| LD | `nn.Linear(10, 1)` | 10 | 1 | Identity | Outputs AGO-space prediction ŷ_ago |

**Why LB dim = 3**: The first neuron encodes the 6 grey parameters (a + 5 b's) via its 5 input weights + 1 bias. The remaining 2 neurons provide supplementary feature transformation capacity before the sigmoid hidden layer.

### Grey coefficient extraction (Eq. 6)

LB is `nn.Linear(5, 3, bias=True)`. After training, extract from the first LB neuron:

```python
w = model.LB.weight[0]  # shape (5,)
b = model.LB.bias[0]    # scalar
w11 = w[0]

a  = -w11                           # development coefficient
b1 = w[1] / w11                     # voltage influence
b2 = w[2] / w11                     # humidity influence
b3 = w[3] / w11                     # hydrogen pressure influence
b4 = w[4] / w11                     # temperature influence
b5 = b  / w11                       # current influence (from bias)
```

This gives exactly 1 a + 5 b_i = 6 grey parameters from 5 weights + 1 bias. This matches the paper: `a=-w11`, `b_i=w_{1,i+1}/w11` (i=1..5), with b5 encoded in the bias.

### Training forward pass

```python
def forward(x: Tensor) -> Tensor:
    # x: (batch, 5) — normalized features at each characterization point
    x = self.LA(x)       # Identity passthrough, (batch, 5)
    x = self.LB(x)       # Linear → (batch, 3)
    x = torch.sigmoid(self.LC(x))  # Linear + Sigmoid → (batch, 10)
    x = self.LD(x)       # Linear → (batch, 1), AGO-space prediction
    return x.squeeze(-1) # (batch,)
```

### Prediction (post-training analytic path)

After training, prediction does **not** use `forward()`. Instead:

```python
def predict(self, data: dict) -> dict:
    a, bs = self.extract_grey_coeffs()  # a: float, bs: (5,) array
    x0 = data["X_all"][0, 0]            # first point's normalized voltage (x^(0)(1))
    k_vals = np.arange(len(data["char_times"]))   # [0, 1, ..., 7]

    y_ago_pred = np.zeros(len(k_vals))
    y_ago_pred[0] = x0                               # initial condition

    for i, k in enumerate(k_vals[1:], start=1):
        # u_i = features at prediction point (all 5, exogenously available)
        u = data["X_all"][i]            # shape (5,)
        u_sum = np.dot(bs, u)           # Σ b_i * u_i
        # Eq. (4) time response
        y_ago_pred[i] = (x0 - u_sum / a) * np.exp(-a * k) + u_sum / a

    # IAGO (Eq. 8)
    y_pred_norm = np.diff(y_ago_pred)   # shape (7,)

    # Inverse normalization
    y_pred = y_pred_norm * (data["target_dmax"] - data["target_dmin"]) + data["target_dmin"]

    return {"y_pred": y_pred, "y_true": data["y_all_raw"][1:], "a": a, "bs": bs}
```

### Weight initialization

All `nn.Linear` layers: `nn.init.uniform_(w, 0.0, 0.5)`. Biases: `nn.init.uniform_(b, 0.0, 0.5)`. Per paper §3.3.

## Training (`train.py`)

### Hyperparameters (from paper §3.3)

| Parameter | Value |
|-----------|-------|
| LC hidden nodes | 10 |
| LB output dim | 3 |
| LC activation | Sigmoid |
| LD activation | Linear (Identity) |
| Optimizer | SGD |
| Learning rate | 0.0002 |
| Weight init range | U(0, 0.5) |
| Max epochs | 1000 |
| Loss function | MSELoss |
| Train/val split | 6/2 |

### Training loop

```python
def train(model, data, config):
    optimizer = SGD(model.parameters(), lr=0.0002)
    loss_fn = MSELoss()
    X = torch.tensor(data["X_train"], dtype=torch.float32)
    y = torch.tensor(data["y_train_ago"], dtype=torch.float32)

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss={loss.item():.6f}")

    return model
```

### Small-sample considerations

- 6 training samples → full batch gradient descent (no mini-batching)
- Use `SGD` (not Adam) to match the paper's gradient descent approach
- Track loss curve; on such small data, expect near-zero training loss within a few hundred epochs
- Support `run(seed, ...)` wrapper for multiple independent runs to get reliable statistics

## Evaluation (`main.py`)

### Metrics

- **MAPE (%)** = mean(|(y_true - y_pred) / y_true|) × 100
- **MSE** = mean((y_true - y_pred)²)
- **RMSE** = sqrt(MSE)

### Outputs

1. Console: final training loss, validation MAPE/MSE/RMSE, extracted grey coefficients (a, b1..b5)
2. Plot 1: predicted vs actual voltage degradation curve over all 8 characterization points
3. Plot 2: training loss over epochs

### Entry point sketch

```python
def main(dataset_name="FC1", seed=42):
    cfg = FC1_CONFIG if dataset_name == "FC1" else FC2_CONFIG
    train_cfg = TRAIN_CONFIG
    data = load_dataset(cfg, seed=seed)
    model = GNN()
    train(model, data, train_cfg)
    results = eval(model, data, train_cfg)
    plot_degradation(results, data)
    plot_loss(results)
    return results
```

## Error Handling

- **File not found**: raise `FileNotFoundError` with the resolved path
- **Mismatched characterization point count**: raise `ValueError` with expected vs actual
- **Division by zero in normalization** (constant column): replace denominator with 1.0
- **a ≈ 0 after training** (grey model singularity): log warning, model unfit for that seed

## Testing Strategy

1. **Unit: normalize roundtrip** — `inverse_normalize(normalize(x)) ≈ x` within float tolerance
2. **Unit: AGO/IAGO roundtrip** — `iago_sequence(ago_sequence(y)) ≈ y[1:]` within float tolerance
3. **Unit: grey coefficient extraction** — construct a `GNN` with known LB weights, verify `extract_grey_coeffs()` returns expected a, b_i
4. **Integration: data pipeline shapes** — `load_dataset(FC1_CONFIG)` returns `X_train(6,5)`, `y_train_ago(6,)`, `X_val(2,5)`, `y_val_ago(2,)`
5. **Integration: full pipeline on FC1** — train + eval, assert MAPE < 2% (standard GNN baseline; paper's PSO-GNN achieves ~1.3%)
