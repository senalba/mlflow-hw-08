import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict

# ---------------------------
# Training utilities
# ---------------------------


def train_pytorch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_activation: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: str = "cpu",
    patience: int = 5,
) -> Dict[str, list]:
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if output_activation == "linear"
        else torch.nn.BCELoss()
    )
    history = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_mae = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).squeeze()
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_mae.append(torch.abs(preds - yb).mean().item())
        model.eval()
        val_losses, val_mae = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).squeeze()
                val_preds = model(xb).squeeze()
                val_loss = criterion(val_preds, yb)
                val_losses.append(val_loss.item())
                val_mae.append(torch.abs(val_preds - yb).mean().item())
        epoch_loss = np.mean(train_losses)
        epoch_mae = np.mean(train_mae)
        val_epoch_loss = np.mean(val_losses)
        val_epoch_mae = np.mean(val_mae)
        history["loss"].append(epoch_loss)
        history["mae"].append(epoch_mae)
        history["val_loss"].append(val_epoch_loss)
        history["val_mae"].append(val_epoch_mae)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f} - "
            f"Val Loss: {val_epoch_loss:.4f} - Val MAE: {val_epoch_mae:.4f}"
        )
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            model.load_state_dict(best_model_state)
            break
    return history
