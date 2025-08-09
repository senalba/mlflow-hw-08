import json
import boto3
import torch
import mlflow.pytorch

# --- Configuration ---

S3_BUCKET = "va-titatic"

S3_KEY = "models/pytorch-model"

# Internal model path (in MLflow artifact)
MLFLOW_MODEL_PATH = "model"

# Required input features
FEATURES = ["Pclass", "Age", "Siblings/Spouses Aboard", "Parch", "Fare"]

s3 = boto3.client("s3")

# --- Model Loader ---


def download_model_from_s3(bucket, key):
    model_uri = f"s3://{bucket}/{key}"
    model = mlflow.pytorch.load_model(model_uri)
    return model


# Cache the model for warm Lambda invocations
model = None

# --- Lambda Handler ---


def handler(event, context):
    global model

    if model is None:
        model = download_model_from_s3(S3_BUCKET, S3_KEY)

    try:
        # Parse input JSON
        input_data = json.loads(event["body"]) if "body" in event else event

        # Convert to tensor
        input_tensor = torch.tensor(
            [[float(input_data[feature]) for feature in FEATURES]], dtype=torch.float32
        )

        # Run inference
        with torch.no_grad():
            prediction = model(input_tensor).item()

        predicted_label = int(prediction >= 0.5)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"probability": prediction, "prediction": predicted_label}
            ),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
