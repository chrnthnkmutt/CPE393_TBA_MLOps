import pickle
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_pipeline.utils import add_interactions, add_custom_features


# ─── Load the model ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "production_models", "lightgbm_pipeline.pkl")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ─── Raw input example ──────────────────────────────────────────────────
data = {
    "age": 28,
    "workclass": "Self-emp-not-inc",
    "education": "Bachelors",
    "education.num": 13,
    "marital.status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital.gain": 10000,
    "capital.loss": 0,
    "hours.per.week": 45,
    "native.country": "United-States"
}

df = pd.DataFrame([data])

# ─── Apply custom features if your model expects them ───────────────────
df = add_custom_features(df)

# ─── Make the prediction ─────────────────────────────────────────────────────
y_pred = model.predict(df)[0]
y_proba = model.predict_proba(df)[0]

# ─── Display results ──────────────────────────────────────────────────
print("✅ Prediction:", y_pred)
print("📊 Probabilities:", y_proba)
print("📚 Classes:", model.classes_)
