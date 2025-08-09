import os
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import boto3
import mlflow

# ---------------------------
# Main MLOps pipeline
# ---------------------------


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, nk))
        else:
            flat[nk] = v
    return flat


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
        self.features = [
            "Pclass",
            "Age",
            "Siblings/Spouses Aboard",
            "Parents/Children Aboard",
            "Fare",
        ]
        # Let env override the tracking URI
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_uri)

        # AWS creds for pandas+s3fs/boto3 in this container
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region

        self.s3 = boto3.client("s3")

        # ---- MLflow setup
        mlflow.set_tracking_uri(self.mlflow_uri)
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Titanic")
        mlflow.set_experiment(experiment_name)

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
        train_strategy: callable that returns (model, metrics_dict, signature, input_example)
        """
        if not log_to_mlflow:
            model, metrics, signature, input_example = train_strategy(
                df, **strategy_params
            )
            return model, metrics

        # name the run so it’s easy to spot
        # with mlflow.start_run(run_name=model_name or "train"):
        with mlflow.start_run(run_name=model_name or "train", nested=True):
            model, metrics, signature, input_example = train_strategy(
                df, **strategy_params
            )

            # ---- Params & tags
            if strategy_params:
                mlflow.log_params(_flatten(strategy_params))
            mlflow.set_tags(
                {
                    "model_name": model_name or "model",
                    "features": ",".join(self.features),
                    "dataset_uri": f"s3://{self.bucket_name}/{self.s3_prefix}",
                }
            )

            # ---- Metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # ---- Logged model (MLflow 3.x "name=" flow)
            if model_name and "RandomForest" in model_name:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name,
                    signature=signature,
                    input_example=input_example,
                    await_registration_for=None,
                )
            elif model_name == "PyTorchMLP":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name=model_name,
                    signature=signature,
                    input_example=input_example,
                    await_registration_for=None,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=model_name or "model",
                    signature=signature,
                    input_example=input_example,
                    await_registration_for=None,
                )

            return model, metrics
    def run(self, train_file, strategies):
        df_train = self.load_csv_from_s3(train_file)
        df_train = self.preprocess(df_train)

        # parent run to group all strategies
        with mlflow.start_run(run_name="titanic-exp"):
            # log dataset once at the parent level (see #2)
            mlflow.log_input(
                mlflow.data.from_pandas(
                    df_train[self.features + ["Survived"]],
                    source=f"s3://{self.bucket_name}/{self.s3_prefix}/{train_file}",
                    name="titanic_train",
                ),
                context="training",
            )

            results = {}
            for name, (strategy, params) in strategies.items():
                print(f"\n=== Training with strategy: {name} ===")
                # tell child runs they’re nested
                model, metrics = self.train(
                    df_train, strategy, params, model_name=name,  # train() already starts a run
                )
                results[name] = (model, metrics)
            return results
    # def run(
    #     self, train_file: str, strategies: Dict[str, Tuple[Callable, Dict[str, Any]]]
    # ):
    #     df_train = self.load_csv_from_s3(train_file)
    #     df_train = self.preprocess(df_train)
    #     results = {}
    #     for name, (strategy, params) in strategies.items():
    #         print(f"\n=== Training with strategy: {name} ===")
    #         model, metrics = self.train(df_train, strategy, params, model_name=name)
    #         results[name] = (model, metrics)
    #     return results
