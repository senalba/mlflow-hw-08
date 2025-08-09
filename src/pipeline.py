import os
import json 
import tempfile
from joblib import dump

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
        self.medians = {}
        for col in self.features:
            m = float(df[col].median())
            self.medians[col] = m
            if df[col].isnull().any():
                df[col] = df[col].fillna(m)
        return df

    def train(
        self,
        df: pd.DataFrame,
        train_strategy: Callable,
        strategy_params: Dict[str, Any],
        log_to_mlflow: bool = True,
        model_name: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if not log_to_mlflow:
            model, metrics, signature, input_example = train_strategy(
                df, **strategy_params
            )
            return model, metrics

        # name the run so it’s easy to spot
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
                try:
                    # Export serving bundle for Lambda
                    # Note: this is a workaround for the fact that sklearn models
                    # cannot be directly used in Lambda without preprocessing.
                    # We create a simple preprocessing step that fills missing values
                    # with the median of each feature.
                    # This is not ideal, but it works for this example.
                    # recompute medians from the df used for training (OK after fillna)
                    medians = {c: float(df[c].median()) for c in self.features}
                    serving_prefix = os.getenv("SERVING_S3_PREFIX", "serving/random_forest")

                    with tempfile.TemporaryDirectory() as d:
                        model_path = os.path.join(d, "model.pkl")
                        prep_path = os.path.join(d, "preprocess.json")

                        dump(model, model_path)
                        with open(prep_path, "w", encoding="utf-8") as f:
                            json.dump({"features": self.features, "medians": medians}, f, ensure_ascii=False)

                        # 1) keep a copy in this MLflow run
                        mlflow.log_artifact(model_path, artifact_path="serving_bundle")
                        mlflow.log_artifact(prep_path, artifact_path="serving_bundle")

                        # 2) upload the bundle to S3 for Lambda
                        self.s3.upload_file(model_path, self.bucket_name, f"{serving_prefix}/model.pkl")
                        self.s3.upload_file(prep_path, self.bucket_name, f"{serving_prefix}/preprocess.json")

                    mlflow.set_tags({
                        "lambda_model_bucket": self.bucket_name,
                        "lambda_model_prefix": serving_prefix,
                    })
                    print(f"Exported serving bundle to s3://{self.bucket_name}/{serving_prefix}/")
                except Exception as e:
                    print(f"Failed to export serving bundle for Lambda: {e}")
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
