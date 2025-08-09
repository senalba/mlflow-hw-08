# Titanic MLOps Pipeline with AWS, MLflow, and Docker

## 🧭 Overview

This project provides a complete, end-to-end MLOps pipeline for the classic Titanic survival prediction problem. It demonstrates best practices for building a modular, scalable, and reproducible machine learning system using modern tools.

The pipeline fetches the Titanic dataset, preprocesses it, trains multiple models (a scikit-learn RandomForest and a PyTorch MLP), tracks experiments with MLflow, and stores artifacts in AWS S3. It is designed to be run in a containerized environment using Docker. Additionally, it includes an optional AWS Lambda function for serving model inference.

## ✨ Core Components

*   **Data Management (AWS S3)**: The Titanic dataset is stored in and loaded from an S3 bucket, ensuring a single source of truth and decoupling data from the training code.
*   **Training Pipeline (`src/pipeline.py`)**: A flexible, dependency-injected pipeline that handles data loading, preprocessing, and model training. It is designed to accept different training strategies.
*   **Model Training (`src/strategies.py`)**: Implements two distinct training strategies:
    *   A `RandomForestClassifier` from scikit-learn.
    *   A multi-layer perceptron (MLP) using PyTorch.
*   **Experiment Tracking (MLflow)**: All training runs, parameters, metrics, and model artifacts are logged to an MLflow Tracking Server, which can be run locally via Docker Compose.
*   **Containerization (Docker)**: The entire environment, including the MLflow server and the training pipeline, is containerized with Docker and orchestrated with Docker Compose for easy setup and execution.
*   **Inference (AWS Lambda)**: An optional Lambda function (`lambda/handler.py`) is provided to demonstrate how a trained model could be deployed for serverless inference.

---

## 🖼️ Screenshots

Here is a screenshot of the MLflow UI showing the experiment runs:

![MLflow Experiments](screenshots/Screenshot%202025-08-08%20at%2019.49.00.png)

And here is a view of the artifacts logged for a specific run:

![MLflow Artifacts](screenshots/Screenshot%202025-08-08%20at%2019.49.36.png)

---

## 🗂️ Project Structure

```
/Users/valba/Documents/mlflow-hw-08/
├───.dockerignore
├───.gitignore
├───docker-compose.yml
├───Dockerfile
├───Dockerfile.mlflow
├───LICENSE
├───README.md
├───requirements.txt
├───.git/...
├───docs/
│   └───setup_guide.md
├───lambda/
│   ├───handler.py
│   └───requirements.txt
├───mlflow_db/
├───mlruns/...
├───screenshots/
└───src/
    ├───__init__.py
    ├───main.py
    ├───model.py
    ├───pipeline.py
    ├───strategies.py
    ├───titanic.csv
    ├───train_utils.py
    └───__pycache__/
```

---

## 🔧 Setup Instructions

### 1. AWS Configuration

Set up a free-tier AWS account and create an S3 bucket (e.g., `va-titanic`).
Add credentials to a `.env` file in the root directory:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

### 2. Upload Dataset to S3

Artifacts appear in:
`s3://$MLFLOW_ARTIFACTS_BUCKET/mlflow-artifacts/...`

Dataset is stored at:
`s3://$MLFLOW_ARTIFACTS_BUCKET/$DATA_S3_PREFIX/titanic.csv`
---

## 🚀 Usage

### Running the Training Pipeline

To run the entire pipeline, including the MLflow Tracking Server, use Docker Compose:

```bash
docker-compose up --build
```

This command will:
1.  Build the Docker images for the MLflow server and the training environment.
2.  Start the MLflow server, which will be accessible at `http://localhost:5000`.
3.  Run the training pipeline defined in `src/main.py`. The script will:
    *   Ensure the Titanic dataset is in your S3 bucket.
    *   Train both the RandomForest and PyTorch MLP models.
    *   Log all experiments, parameters, and metrics to the MLflow server.


## AWS Lambda Inference (optional)

This repo includes a minimal **serverless inference** path for the RandomForest model using **AWS Lambda**.

### What training exports
During training, the pipeline writes a lightweight “serving bundle” to S3:
- `model.pkl` – the trained **scikit-learn** RandomForest
- `preprocess.json` – metadata for inference (feature order + medians for imputation)

S3 location:
`s3://$MODEL_BUCKET/$MODEL_BASE/`
e.g. `s3://mlflow-vasyl-a-titanic/serving/random_forest/`

 Configure via env:
> - `MODEL_BUCKET` (defaults to `MLFLOW_ARTIFACTS_BUCKET`)
> - `MODEL_BASE` (defaults to `serving/random_forest`)

### Lambda implementation
- Code: `lambda/handler.py`
- Container image: `Dockerfile.lambda` (base: `public.ecr.aws/lambda/python:3.10`)
- **Version pins** (match trainer to avoid pickle/ABI issues):  
  `boto3==1.34.162`, `joblib==1.5.1`, `numpy==2.2.6`, `scipy==1.15.3`, `scikit-learn==1.7.1` (in `lambda/requirements.txt`)

### Local testing (via Docker Compose)
1) Ensure the bundle exists:
```bash
> aws s3 ls "s3://$MODEL_BUCKET/$MODEL_BASE/"
2025-08-08 22:07:05     564809 model.pkl
2025-08-08 22:07:06        215 preprocess.json
```
2) Invoke locally:
```bash
❯ curl -s http://localhost:9000/2015-03-31/functions/function/invocations -d '{"instances":[{"Pclass":3,"Age":22,"Siblings/Spouses Aboard":1,"Parents/Children Aboard":0,"Fare":7.25}]}' 
{"predictions": [0], "probabilities": [0.22821913745252861]}%
```