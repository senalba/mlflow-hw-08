import os, json, tempfile
import boto3, joblib, numpy as np

S3 = boto3.client("s3")
BUCKET = os.environ["MODEL_BUCKET"]
BASE = os.environ.get("MODEL_BASE", "serving/random_forest")

_model = None
_prep = None


def _load():
    global _model, _prep
    if _model is None:
        with tempfile.TemporaryDirectory() as d:
            mp, pp = f"{d}/model.pkl", f"{d}/preprocess.json"
            S3.download_file(BUCKET, f"{BASE}/model.pkl", mp)
            S3.download_file(BUCKET, f"{BASE}/preprocess.json", pp)
            _model = joblib.load(mp)
            with open(pp) as f:
                _prep = json.load(f)
    return _model, _prep


def lambda_handler(event, context):
    model, prep = _load()
    feats = prep["features"]
    med = prep["medians"]

    # Expect: {"instances": [ {feature: value, ...}, ... ]}
    rows = event.get("instances") or []
    X = [[(row.get(c, med.get(c))) for c in feats] for row in rows]
    X = np.array(X, dtype=np.float32)

    # RF: use predict_proba
    probs = model.predict_proba(X)[:, 1].tolist()
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {"predictions": preds, "probabilities": probs}
