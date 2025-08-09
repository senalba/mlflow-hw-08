FROM python:3.10-slim

ARG MLFLOW_VERSION=3.2.0

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" \
    pandas numpy scikit-learn \
    boto3 s3fs python-dotenv pyyaml \
    "torch; sys_platform != 'darwin' or platform_machine != 'arm64'"

ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/src

CMD ["python", "main.py"]
