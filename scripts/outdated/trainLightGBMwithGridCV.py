import os
import warnings
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb

# from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, StandardScaler, OneHotEncoder
)
import category_encoders as ce
import matplotlib.pyplot as plt

# ─── 1. MLflow setup ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
mlflow.set_experiment("Adult_LightGBM_Pipeline")

# ─── 2. Chargement et nettoyage de base ──────────────────────────────────────
raw = pd.read_csv("data/adult.csv")

# remplacer les '?' par la modalité la plus fréquente
for col in ["workclass", "occupation", "native.country"]:
    raw.loc[raw[col].str.strip() == "?", col] = raw[col].mode()[0]

# cible binaire
raw["income"] = raw["income"].str.strip()
raw["target"] = raw["income"].apply(lambda x: 1 if ">50K" in x else 0)


# ─── 3. Fonction d’ingénierie avancée ────────────────────────────────────────
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # flags
    df["has_cap_gain"] = (df["capital.gain"] > 0).astype(int)
    df["has_cap_loss"] = (df["capital.loss"] > 0).astype(int)
    # ratios
    df["gain_loss_ratio"] = (df["capital.gain"] + 1) / (df["capital.loss"] + 1)
    df["work_rate"] = df["hours.per.week"] / (df["age"] + 1)
    # binning âge
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 30, 50, 100],
        labels=["young", "mid", "senior"]
    ).astype(str)
    return df


# ─── 4. Pipeline de features ─────────────────────────────────────────────────
def build_feature_pipeline():
    # colonnes originales
    num_feats      = ["age", "hours.per.week", "education.num",
                      "capital.gain", "capital.loss", "fnlwgt"]
    cat_onehot     = ["workclass", "education", "marital.status",
                      "relationship", "race", "sex", "age_bin"]
    cat_target_enc = ["occupation", "native.country"]
    flag_feats     = ["has_cap_gain", "has_cap_loss"]
    ratio_feats    = ["gain_loss_ratio", "work_rate"]

    # pipeline numérique : log1p ➔ standard scaler ➔ interactions simples
    num_pipeline = Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("interactions", FunctionTransformer(
            lambda X: np.hstack([
                X,
                (X[:, [0]] * X[:, [1]]),               # age * hours.per.week
                (X[:, [2]] / (X[:, [3]] + 1)),         # educ.num / (cap.gain +1)
            ]),
            validate=False
        )),
    ])

    # one-hot pour petites cat
    onehot_pipeline = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="if_binary"
        ))
    ])

    # target‐encoding pour catégories plus “lourdes”
    target_encoder = ce.TargetEncoder(
        cols=cat_target_enc,
        smoothing=0.3
    )

    preprocessor = ColumnTransformer([
        ("num",    num_pipeline,      num_feats + ratio_feats + flag_feats),
        ("onehot", onehot_pipeline,   cat_onehot),
        ("targ",   target_encoder,    cat_target_enc),
    ], remainder="drop")

    return preprocessor


# ─── 5. Construction du pipeline complet ─────────────────────────────────────
preprocessor = build_feature_pipeline()
model = lgb.LGBMClassifier(
    class_weight="balanced",
    device="gpu",              
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
)
pipeline = ImbPipeline([
    ("custom_feat", FunctionTransformer(add_custom_features, validate=False)),
    ("preproc",     preprocessor),
    ("smote",       SMOTE(random_state=42)),
    ("clf",         model),
])


# ─── 6. Split train/test ─────────────────────────────────────────────────────
X = raw.drop(columns=["income", "target"])
y = raw["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── 7. Hyper-param search + éval final ─────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# 1) grille à tester
param_grid = {
    "clf__n_estimators":   [100, 200, 500],
    "clf__learning_rate":  [0.01, 0.05, 0.1],
    "clf__num_leaves":     [31, 63, 127],
    "clf__max_depth":      [-1, 10, 20],
    "clf__subsample":      [0.7, 1.0],
    "clf__colsample_bytree":[0.7, 1.0],
}

# 2) validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3) GridSearchCV sur votre pipeline
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",      # ou "roc_auc" selon votre critère métier
    n_jobs=-1,
    verbose=2
)

# ─── 8. Entraînement + logging MLflow ────────────────────────────────────────
with mlflow.start_run(run_name="LightGBM_with_advanced_features"):
    # pipeline.fit(X_train, y_train)
    search.fit(X_train, y_train)

    # prédictions / proba
    
    # y_pred  = pipeline.predict(X_test)
    # y_proba = pipeline.predict_proba(X_test)[:, 1]
    best_pipeline = search.best_estimator_
    y_pred  = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # calcul des métriques
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    
    print(f"Best params: {search.best_params_}")
    print(f"Best score: {search.best_score_}")  
    print(f"Metrics: Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}, AUC: {auc}")

    # log metrics & modèle
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("auc",       auc)
    # mlflow.sklearn.log_model(pipeline, "lightgbm_pipeline")
    mlflow.sklearn.log_model(best_pipeline, "lightgbm_pipeline")


    # log hyper‐params
    mlflow.log_param("n_estimators",   best_pipeline.n_estimators)
    mlflow.log_param("learning_rate",  best_pipeline.learning_rate)
    mlflow.log_param("num_leaves",     best_pipeline.num_leaves)

    # ── 8. Analyse de seuil ─────────────────────────────────────────────
    thresholds = np.linspace(0.1, 0.9, 17)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred_t = (y_proba > t).astype(int)
        f1s.append(f1_score(y_test, y_pred_t))
        precs.append(precision_score(y_test, y_pred_t))
        recs.append(recall_score(y_test, y_pred_t))

    best_idx = int(np.argmax(f1s))
    best_t   = float(thresholds[best_idx])
    mlflow.log_metric("best_threshold", best_t)

    # tracer & enregistrer la courbe
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1s,  label="F1")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs,  label="Recall")
    plt.axvline(best_t, color="gray", linestyle="--", label=f"Threshold={best_t:.2f}")
    plt.title("Seuil décision LightGBM")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/LightGBM_threshold.png")
    mlflow.log_artifact("models/LightGBM_threshold.png")

print("✅ Entraînement terminé, artefacts disponibles dans mlruns/ …")
