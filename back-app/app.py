from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)


with open('RF_model_prod.pkl', 'rb') as f:
    model = pickle.load(f)


with open("features.json") as f:
    feature_info = json.load(f)
    feature_names = [f["name"] for f in feature_info]


# curl http://localhost:5000/health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# curl http://localhost:5000/features
@app.route("/features", methods=["GET"])
def get_features():
    return jsonify(feature_info)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_vector = {name: data.get(name, 0) for name in feature_names}
    df = pd.DataFrame([input_vector])
    prediction = model.predict(df)[0]
    label = ">50K" if prediction else "<=50K"
    return jsonify({"prediction": label})

@app.route("/predict_proba", methods=["POST"])
def predict_proba():
    data = request.get_json()
    input_vector = {name: data.get(name, 0) for name in feature_names}
    df = pd.DataFrame([input_vector])
    proba = model.predict_proba(df)[0]
    return jsonify(dict(zip(model.classes_.astype(str), proba.tolist())))


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": type(model).__name__,
        "training_date": "2025-05-23",
        "n_features": len(feature_names),
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None)
    })


@app.route("/explain", methods=["POST"])
def explain():
    importances = dict(zip(feature_names, model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    return jsonify(sorted_importances)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
