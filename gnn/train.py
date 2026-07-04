"""GNN 训练循环。

小样本 (6 个训练点) → 全批量梯度下降, Adam 优化器。
forward 需要 k (时间步索引), u (特征), x0 (AGO 初始条件)。
"""

import torch
import torch.nn as nn
from gnn.model import GNN
from gnn.config import TrainConfig


def train(
    model: GNN,
    data: dict,
    config: TrainConfig,
    verbose: bool = True,
) -> "tuple[GNN, list[float]]":
    """训练 GNN 模型。

    Args:
        model: 未训练的 GNN 实例。
        data: load_dataset 返回的字典。
        config: 训练超参数。
        verbose: 是否每 100 epoch 打印 loss。

    Returns:
        model: 训练后的 GNN (in-place 修改 + 返回)。
        losses: 每个 epoch 的训练 loss 列表。
    """
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    # 训练数据
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)   # (6, 5)
    y_train = torch.tensor(data["y_train_ago"], dtype=torch.float32) # (6,)

    # k: 训练样本对应的时间步索引 (0-based)
    # 时间顺序划分: 训练集是前 6 个点, k = [0, 1, 2, 3, 4, 5]
    k_train = torch.arange(len(y_train), dtype=torch.float32)      # (6,)

    # x0: AGO 初始条件 x^(0)(1) = 第一个归一化电压值
    x0 = torch.tensor(data["y_all_ago"][0], dtype=torch.float32)   # scalar

    losses = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        y_pred = model(k_train, X_train, x0)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}/{config.epochs}  loss = {loss.item():.8f}")

    if verbose:
        print(f"  epoch {config.epochs:4d}/{config.epochs}  loss = {losses[-1]:.8f}  [done]")

    return model, losses
