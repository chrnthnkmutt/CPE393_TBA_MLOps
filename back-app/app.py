from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_model_path = os.path.join(BASE_DIR, "production_models", "RF_model_prod.pkl")
GBM_model_path = os.path.join(BASE_DIR, "production_models", "GBM_model_prod.pkl")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


with open(RF_model_path, 'rb') as f:
    model = pickle.load(f)


with open("features.json") as f:
    feature_info = json.load(f)
    feature_names = [f["name"] for f in feature_info]
    

# curl http://localhost:5000/api/health
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# curl http://localhost:5000/api/features
@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify(feature_info)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Check for missing features
    missing = [name for name in feature_names if name not in data]
    if missing:
        print(f"⚠️ Missing data. Filling with 0 : {missing}")

    # Create a vector with missing values replaced by 0
    input_vector = {name: data.get(name, 0) for name in feature_names}
    df = pd.DataFrame([input_vector])

    prediction = model.predict(df)[0]
    label = ">50K" if prediction else "<=50K"
    return jsonify({"prediction": label, "missing_features": missing})


@app.route("/api/predict_proba", methods=["POST"])
def predict_proba():
    data = request.get_json()
    missing = [name for name in feature_names if name not in data]

    input_vector = {name: data.get(name, 0) for name in feature_names}
    df = pd.DataFrame([input_vector])
    proba = model.predict_proba(df)[0]

    return jsonify({
        "probabilities": dict(zip(model.classes_.astype(str), proba.tolist())),
        "missing_features": missing
    })



@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": type(model).__name__,
        "training_date": "2025-05-23",
        "n_features": len(feature_names),
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None)
    })


@app.route("/api/explain", methods=["POST"])
def explain():
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_names, model.feature_importances_))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        return jsonify(sorted_importances)
    else:
        return jsonify({"error": "This model does not provide feature importances."}), 400

# curl http://localhost:5000/api/metrics
@app.route("/api/metrics", methods=["GET"])
def metrics():
    with open('RF_metrics.json', 'r') as f:
        metrics = json.load(f)
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





# "age": z.number().min(17).max(90),
#     "capital.gain": z.number().min(0).max(99999),
#     "capital.loss": z.number().min(0).max(4356),
#     "hours.per.week": z.number().min(1).max(99),
#     // "hours-per-week": z.number().min(1).max(99),
    
  
#     "education_Assoc-acdm": z.boolean(),
#     "education_Assoc-voc": z.boolean(),
#     "education_Bachelors": z.boolean(),
#     "education_Doctorate": z.boolean(),
#     "education_HS-grad": z.boolean(),
#     "education_Masters": z.boolean(),
#     "education_Prof-school": z.boolean(),
  
#     "marital.status_Married": z.boolean(),
#     "marital.status_Never-married": z.boolean(),
#     "marital.status_Separated": z.boolean(),
#     "marital.status_Widowed": z.boolean(),
  
#     "occupation_Adm-clerical": z.boolean(),
#     "occupation_Armed-Forces": z.boolean(),
#     "occupation_Craft-repair": z.boolean(),
#     "occupation_Exec-managerial": z.boolean(),
#     "occupation_Farming-fishing": z.boolean(),
#     "occupation_Handlers-cleaners": z.boolean(),
#     "occupation_Machine-op-inspct": z.boolean(),
#     "occupation_Priv-house-serv": z.boolean(),
#     "occupation_Prof-specialty": z.boolean(),
#     "occupation_Protective-serv": z.boolean(),
#     "occupation_Sales": z.boolean(),
#     "occupation_Tech-support": z.boolean(),
#     "occupation_Transport-moving": z.boolean(),
  
#     "race_Amer-Indian-Eskimo": z.boolean(),
#     "race_Asian-Pac-Islander": z.boolean(),
#     "race_Other": z.boolean(),
#     "race_White": z.boolean(),
  
#     "relationship_Husband": z.boolean(),
#     "relationship_Not-in-family": z.boolean(),
#     "relationship_Other-relative": z.boolean(),
#     "relationship_Own-child": z.boolean(),
#     "relationship_Unmarried": z.boolean(),
#     "relationship_Wife": z.boolean(),
  
#     "sex_Female": z.boolean(),
  
#     "workclass_Govt_employees": z.boolean(),
#     "workclass_Never-worked": z.boolean(),
#     "workclass_Private": z.boolean(),
#     "workclass_Self_employed": z.boolean(),
#     "workclass_Without-pay": z.boolean()
