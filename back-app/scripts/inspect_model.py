import pickle
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_model_path = os.path.join(BASE_DIR, "..", "production_models", "RF_model_20250529_075712.pkl")
GBM_model_path = os.path.join(BASE_DIR, "..", "production_models", "GBM_model_20250529_075710.pkl")

def inspect_model(model, model_name):
    print(f"\n{'='*50}")
    print(f"🔍 INSPECTING MODEL {model_name}")
    print(f"{'='*50}")

    print("\nMODEL TYPE")
    print(type(model))

    print("\nOUTPUT CLASSES")
    if hasattr(model, "classes_"):
        print("classes_ :", model.classes_)

    print("\nSCORE (if calculated with .score())")
    try:
        X_fake = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)
        print("Score on 1 fake sample :", model.score(X_fake, [model.classes_[0]]))
    except Exception as e:
        print("No score available :", e)

    print("\nUSED FEATURES")
    if hasattr(model, "feature_names_in_"):
        print("feature_names_in_ :", list(model.feature_names_in_))

    print("\nFEATURE IMPORTANCES")
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(model.feature_names_in_, model.feature_importances_))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        print("Top 10 most important features :", list(sorted_importances.items())[:10])

    print("\nHYPERPARAMETERS")
    if hasattr(model, "get_params"):
        print("Model parameters :", model.get_params())

with open(RF_model_path, "rb") as f:
    rf_model = pickle.load(f)
    inspect_model(rf_model, "RANDOM FOREST")

with open(GBM_model_path, "rb") as f:
    gbm_model = pickle.load(f)
    inspect_model(gbm_model, "GRADIENT BOOSTING")
