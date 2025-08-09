from dotenv import load_dotenv
import os
import botocore
import boto3

from pipeline import MLOpsTitanicPipeline
from strategies import sklearn_rf_strategy, pytorch_mlp_strategy

load_dotenv()

DATA_BUCKET = os.getenv("MLFLOW_ARTIFACTS_BUCKET", "mlflow-vasyl-a-titanic")
S3_PREFIX = os.getenv("DATA_S3_PREFIX", "dataset")
DATASET_LOCAL = os.getenv("DATASET_LOCAL_PATH", "titanic.csv")
DATASET_KEY = os.getenv("DATASET_KEY", "titanic.csv")


def ensure_dataset_in_s3():
    s3 = boto3.client("s3")
    key = f"{S3_PREFIX}/{DATASET_KEY}"
    try:
        s3.head_object(Bucket=DATA_BUCKET, Key=key)
        print(f"✅ Dataset already in s3://{DATA_BUCKET}/{key}")
    except botocore.exceptions.ClientError as e:
        if e.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        ) == 404 or e.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            print(f"⬆️  Uploading {DATASET_LOCAL} → s3://{DATA_BUCKET}/{key}")
            s3.upload_file(DATASET_LOCAL, DATA_BUCKET, key)
            print("✅ Upload complete")
        else:
            raise


ensure_dataset_in_s3()


pipeline = MLOpsTitanicPipeline(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    bucket_name=DATA_BUCKET,
    s3_prefix=S3_PREFIX,
    mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
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

if __name__ == "__main__":
    print("starting pipeline.run")
    pipeline.run(DATASET_KEY, strategies)
    print("✅ finished pipeline.run")
