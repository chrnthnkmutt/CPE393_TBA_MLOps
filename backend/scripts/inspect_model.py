import pickle
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml_pipeline.utils import add_custom_features, add_interactions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "production_models", "lightgbm_pipeline.pkl")

def inspect_model(model, model_name):
    print(f"\n{'='*50}")
    print(f"🔍 INSPECTING MODEL {model_name}")
    print(f"{'='*50}")

    print("\nMODEL TYPE")
    print(type(model))

    clf = model.named_steps["clf"]

    print("\nOUTPUT CLASSES")
    if hasattr(clf, "classes_"):
        print("classes_ :", clf.classes_)

    print("\nSCORE (if calculated with .score())")
    try:
        # Simulation d'un input avec les bons features
        X_fake = pd.DataFrame(np.zeros((1, len(clf.feature_names_in_))), columns=clf.feature_names_in_)
        print("Score on 1 fake sample :", model.score(X_fake, [clf.classes_[0]]))
    except Exception as e:
        print("No score available :", e)

    print("\nUSED FEATURES")
    if hasattr(clf, "feature_names_in_"):
        print("feature_names_in_ :", list(clf.feature_names_in_))

    print("\nFEATURE IMPORTANCES")
    try:
        importances = clf.feature_importances_
        importances_dict = dict(zip(clf.feature_names_in_, importances))
        sorted_importances = dict(sorted(importances_dict.items(), key=lambda x: x[1], reverse=True))
        print("Top 10 most important features :", list(sorted_importances.items())[:10])
    except Exception as e:
        print("Feature importances not available:", e)

    print("\nHYPERPARAMETERS")
    print(clf.get_params())

if __name__ == "__main__":
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        inspect_model(model, "LIGHTGBM")
