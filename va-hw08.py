from dotenv import load_dotenv
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import boto3
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from mlflow.models.signature import infer_signature


load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")


# ---------------------------
# PyTorch TitanicMLP model
# ---------------------------


class TitanicMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=(64, 32),
        dropout_rate=0.5,
        output_activation="sigmoid",
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 1))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "linear":
            pass  # No activation
        else:
            raise ValueError("Unsupported output activation")
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
    criterion = nn.BCELoss() if output_activation == "sigmoid" else nn.MSELoss()
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


# ---------------------------
# Main MLOps pipeline
# ---------------------------


class MLOpsTitanicPipeline:
    """
    Flexible MLOps pipeline for Titanic dataset. Supports arbitrary training strategy via DI.
    """

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str,
        bucket_name: str,
        s3_prefix: str = "dataset",
        mlflow_uri: str = "http://localhost:5000",
    ):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        self.mlflow_uri = mlflow_uri
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        self.s3 = boto3.client("s3")
        mlflow.set_tracking_uri(mlflow_uri)

    def upload_to_s3(self, local_path: str, s3_filename: str) -> None:
        object_name = f"{self.s3_prefix}/{s3_filename}"
        self.s3.upload_file(local_path, self.bucket_name, object_name)
        print(f"Uploaded {local_path} to s3://{self.bucket_name}/{object_name}")

    def load_csv_from_s3(self, s3_filename: str) -> pd.DataFrame:
        object_name = f"{self.s3_prefix}/{s3_filename}"
        s3_url = f"s3://{self.bucket_name}/{object_name}"
        return pd.read_csv(s3_url)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.features + ["Survived"]].copy()
        for col in self.features:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def train(
        self,
        df: pd.DataFrame,
        train_strategy: Callable,
        strategy_params: Dict[str, Any],
        log_to_mlflow: bool = True,
        model_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        train_strategy: callable that returns (model, metrics_dict)
        strategy_params: parameters to pass to train_strategy
        """
        if log_to_mlflow:
            mlflow.start_run()
        try:
            model, metrics, signature, input_example = train_strategy(
                df, **strategy_params
            )
            if log_to_mlflow:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                if (
                    model_name
                    and hasattr(mlflow, "sklearn")
                    and "RandomForest" in model_name
                ):
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )
                elif model_name == "PyTorchMLP":
                    mlflow.pytorch.log_model(model, artifact_path="model")
            return model, metrics
        finally:
            if log_to_mlflow:
                mlflow.end_run()

    def run(
        self, train_file: str, strategies: Dict[str, Tuple[Callable, Dict[str, Any]]]
    ):
        df_train = self.load_csv_from_s3(train_file)
        df_train = self.preprocess(df_train)
        results = {}
        for name, (strategy, params) in strategies.items():
            print(f"\n=== Training with strategy: {name} ===")
            model, metrics = self.train(df_train, strategy, params, model_name=name)
            results[name] = (model, metrics)
        return results


# ---------------------------
# Training strategies
# ---------------------------


def sklearn_rf_strategy(df: pd.DataFrame, **kwargs):
    from sklearn.ensemble import RandomForestClassifier

    features = kwargs.get("features", ["Pclass", "Age", "SibSp", "Parch", "Fare"])
    X = df[features].astype(float)
    y = df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(**kwargs.get("rf_params", {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    from mlflow.models.signature import infer_signature

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:2]
    metrics = {"accuracy": acc, "f1_score": f1}
    return model, metrics, signature, input_example


def pytorch_mlp_strategy(df: pd.DataFrame, **kwargs):
    features = kwargs.get("features", ["Pclass", "Age", "SibSp", "Parch", "Fare"])
    device = kwargs.get("device", "cpu")
    X = df[features].astype(np.float32).values
    y = df["Survived"].astype(np.float32).values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    batch_size = kwargs.get("batch_size", 32)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    input_dim = X.shape[1]
    model = TitanicMLP(input_dim, **kwargs.get("mlp_params", {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))
    history = train_pytorch(
        model,
        optimizer,
        kwargs.get("mlp_params", {}).get("output_activation", "sigmoid"),
        train_loader,
        val_loader,
        kwargs.get("num_epochs", 20),
        device,
        patience=kwargs.get("patience", 5),
    )
    # Validation metrics
    model.eval()
    with torch.no_grad():
        preds = []
        labels = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            labels.append(yb.cpu().numpy())
        preds = np.concatenate(preds).reshape(-1)
        labels = np.concatenate(labels).reshape(-1)
        y_pred_bin = (preds > 0.5).astype(int)
        acc = accuracy_score(labels, y_pred_bin)
        f1 = f1_score(labels, y_pred_bin)
    metrics = {"accuracy": acc, "f1_score": f1}
    signature = None  # Not used for PyTorch logging in MLflow
    input_example = torch.tensor(X_train[:2])
    return model, metrics, signature, input_example


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    pipeline = MLOpsTitanicPipeline(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        bucket_name="va-titanix",
        s3_prefix="dataset",
        mlflow_uri="http://localhost:5000",
    )

    strategies = {
        "RandomForest": (
            sklearn_rf_strategy,
            {"rf_params": {"n_estimators": 100, "max_depth": 6}},
        ),
        "PyTorchMLP": (
            pytorch_mlp_strategy,
            {
                "mlp_params": {
                    "hidden_units": (64, 32),
                    "dropout_rate": 0.5,
                    "output_activation": "sigmoid",
                },
                "lr": 1e-3,
                "num_epochs": 30,
                "batch_size": 32,
                "device": "cpu",
            },
        ),
    }

    pipeline.run("train.csv", strategies)
