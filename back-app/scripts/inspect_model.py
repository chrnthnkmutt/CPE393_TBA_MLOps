import pickle
import pandas as pd
import numpy as np
import os

model_path = "../models/RF_model_20250515_112838.pkl"

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

print("\n🔍 MODEL TYPE")
print(type(model))

print("\n📊 OUTPUT CLASSES")
if hasattr(model, "classes_"):
    print("classes_ :", model.classes_)

print("\n📈 SCORE (if calculated with .score())")
try:
    X_fake = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)
    print("Score on 1 fake sample :", model.score(X_fake, [model.classes_[0]]))
except Exception as e:
    print("No score available :", e)

print("\n🧠 USED FEATURES")
if hasattr(model, "feature_names_in_"):
    print("feature_names_in_ :", list(model.feature_names_in_))

print("\n📌 FEATURE IMPORTANCES")
if hasattr(model, "feature_importances_"):
    importances = dict(zip(model.feature_names_in_, model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("Top 10 most important features :", list(sorted_importances.items())[:10])

print("\n⚙️ HYPERPARAMETERS")
if hasattr(model, "get_params"):
    print("Model parameters :", model.get_params())
