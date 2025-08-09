import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlflow.models.signature import infer_signature

from model import TitanicMLP
from train_utils import train_pytorch



def sklearn_rf_strategy(df, **kwargs):

    features = kwargs.get(
        "features",
        ["Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"],
    )
    X = df[features].astype(float)
    y = df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_params = {"random_state": 42, **kwargs.get("rf_params", {})}
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:2]
    metrics = {"accuracy": acc, "f1_score": f1}
    return model, metrics, signature, input_example



def pytorch_mlp_strategy(df, **kwargs):

    # ✅ correct default features
    features = kwargs.get(
        "features",
        ["Pclass", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"],
    )
    device = kwargs.get("device", "cpu")

    # floats for X, float targets in {0,1}
    X = df[features].astype(np.float32).values
    y = df["Survived"].astype(np.float32).values.reshape(-1, 1)  # <- (N,1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    batch_size = kwargs.get("batch_size", 32)
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    mlp_params = {**kwargs.get("mlp_params", {})}
    # ✅ prefer logits (linear) + BCEWithLogitsLoss
    mlp_params.setdefault("output_activation", "linear")

    model = TitanicMLP(input_dim, **mlp_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))

    # train (your train_utils should use BCEWithLogitsLoss when activation == 'linear')
    history = train_pytorch(
        model,
        optimizer,
        mlp_params.get("output_activation", "linear"),
        train_loader,
        val_loader,
        kwargs.get("num_epochs", 20),
        device,
        patience=kwargs.get("patience", 5),
    )

    # eval
    model.eval()
    with torch.no_grad():
        probs_list, labels_list = [], []
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)                 # (B,1)
            probs = torch.sigmoid(logits)      # ✅ sigmoid for probabilities
            probs_list.append(probs.cpu().numpy())
            labels_list.append(yb.cpu().numpy())
        probs = np.vstack(probs_list).reshape(-1)
        labels = np.vstack(labels_list).reshape(-1)
        y_pred_bin = (probs > 0.5).astype(int)
        acc = accuracy_score(labels, y_pred_bin)
        f1 = f1_score(labels, y_pred_bin)

    # MLflow signature & input example (use numpy, keep on CPU)
    input_example = X_train[:2].astype(np.float32)
    with torch.no_grad():
        ex_logits = model(torch.from_numpy(input_example).to(device))
        ex_probs = torch.sigmoid(ex_logits).cpu().numpy()
    signature = infer_signature(input_example, ex_probs)

    metrics = {"accuracy": acc, "f1_score": f1}
    return model, metrics, signature, input_example

