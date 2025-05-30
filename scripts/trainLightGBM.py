import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
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
import seaborn as sns

# ─── 1. MLflow setup ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
mlflow.set_experiment("Adult_LightGBM_Pipeline")

# ─── 2. Chargement et nettoyage de base ──────────────────────────────────────
raw = pd.read_csv("data/adult.csv")
raw = raw.drop(columns=["fnlwgt"])


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
                    "capital.gain", "capital.loss"]
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
    
    missing_cols = [col for col in num_feats if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the dataset : {missing_cols}")


    preprocessor = ColumnTransformer([
        ("num",    num_pipeline,      num_feats + ratio_feats + flag_feats),
        ("onehot", onehot_pipeline,   cat_onehot),
        ("targ",   target_encoder,    cat_target_enc),
    ], remainder="drop")

    return preprocessor


# ─── 5. Construction du pipeline complet ─────────────────────────────────────
preprocessor = build_feature_pipeline()
model = LGBMClassifier(
    class_weight="balanced",
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

# ─── 6. Form options ─────────────────────────────────────────────────────

# juste après le nettoyage de raw, par exemple avant le train_test_split
categorical_inputs = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country"
]

options = {}
for col in categorical_inputs:
    # tri pour l’UI
    opts = sorted(raw[col].unique().tolist())
    options[col] = opts

# affiche ou sauve en JSON pour ton front
import json
print(json.dumps(options, indent=2, ensure_ascii=False))
with open("models/form_options.json", "w") as f:
    json.dump(options, f, indent=2, ensure_ascii=False)



# ─── 7. Split train/test ─────────────────────────────────────────────────────
X = raw.drop(columns=["income", "target"])
y = raw["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)



# ─── 8. Training + logging MLflow ────────────────────────────────────────
with mlflow.start_run(run_name="LightGBM_with_advanced_features"):
    pipeline.fit(X_tr, y_tr)
    
    lgbm_model = pipeline.named_steps["clf"]
    if hasattr(lgbm_model, "feature_importances_"):
        feature_names = []
        for name, trans, cols in pipeline.named_steps["preproc"].transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "onehot":
                encoder = pipeline.named_steps["preproc"].named_transformers_["onehot"]
                try:
                    feature_names.extend(encoder.get_feature_names_out(cols))
                except:
                    print(f"Error getting feature names for {cols}")
                    feature_names.extend(cols)
            elif name == "targ":
                feature_names.extend(cols)
        
        importances = dict(zip(feature_names, lgbm_model.feature_importances_))
        importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # Afficher toutes les features et leurs importances
        print("\nToutes les features et leurs importances:")
        for feature, importance in importances_sorted.items():
            print(f"{feature}: {importance:.2f}")
        
        # log CSV
        imp_df = pd.DataFrame(importances_sorted.items(), columns=["feature", "importance"])
        imp_df.to_csv("models/feature_importances.csv", index=False)
        mlflow.log_artifact("models/feature_importances.csv")

        # transformer les importances en pourcentage
        imp_df["percentage"] = 100 * imp_df["importance"] / imp_df["importance"].sum()
        
        topn = 20  # top 20 features
        plt.figure(figsize=(10,6))
        plt.barh(imp_df["feature"][:topn], imp_df["percentage"][:topn])
        plt.xlabel("Importance (%)")
        plt.title(f"Top {topn} features - LightGBM")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("models/feature_importance_plot.png")
        mlflow.log_artifact("models/feature_importance_plot.png")
        
    # 1) Prépare ton input_example en float
    X_input_example = X_test.head(1).copy()
    X_input_example = X_input_example.astype({
        col: 'float' for col in X_input_example.select_dtypes('int').columns
    })
    
    print(X_input_example.columns.tolist())
    X_input_example.head(1).to_json("models/example_input.json", orient="records", lines=False)


    # 2) Infère la signature à partir de ce même exemple
    signature = mlflow.models.infer_signature(
        X_input_example,
        pipeline.predict_proba(X_input_example)
    )

    # 3) Log le modèle avec input_example + signature "propres"
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm_pipeline",
        input_example=X_input_example,
        signature=signature
    )

    # log hyper‐params
    mlflow.log_param("n_estimators",   model.n_estimators)
    mlflow.log_param("learning_rate",  model.learning_rate)
    mlflow.log_param("num_leaves",     model.num_leaves)
    
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    # ── 8. Analyse de seuil ─────────────────────────────────────────────
    thresholds = np.linspace(0.1, 0.9, 17)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_val_pred_t = (y_val_proba > t).astype(int)
        f1s.append(f1_score(y_val, y_val_pred_t))
        precs.append(precision_score(y_val, y_val_pred_t))
        recs.append(recall_score(y_val, y_val_pred_t))

    best_idx = int(np.argmax(f1s))
    best_t   = float(thresholds[best_idx])
    mlflow.log_metric("best_threshold", best_t)
    
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred  = (y_test_proba > best_t).astype(int)

    # calcul des métriques
    acc  = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec  = recall_score(y_test, y_test_pred)
    f1   = f1_score(y_test, y_test_pred)
    auc  = roc_auc_score(y_test, y_test_proba)
    
    mlflow.log_metric("final_accuracy", acc)
    mlflow.log_metric("final_precision", prec)
    mlflow.log_metric("final_recall", rec)
    mlflow.log_metric("final_auc", auc)
    mlflow.log_metric("f1_score_best_thresh", f1)

    
    print(f"FinalMetrics: Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}, AUC: {auc}")
    
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8,5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    mlflow.log_artifact("models/confusion_matrix.png")

    # tracer & enregistrer la courbe
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1s,  label="F1")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs,  label="Recall")
    plt.axvline(best_t, color="gray", linestyle="--", label=f"Threshold={best_t:.2f}")
    plt.scatter(best_t, f1s[best_idx], color='red', label=f"F1 max={f1s[best_idx]:.2f}")
    plt.title("LightGBM Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/LightGBM_threshold.png")
    mlflow.log_artifact("models/LightGBM_threshold.png")

print("Training completed, artefacts available in mlruns.")
