from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import json
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_pipeline.utils import add_custom_features, add_interactions


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LIGHTGBM_model_path = os.path.join(BASE_DIR, "production_models", "lightgbm_pipeline.pkl")


MODEL_PATH = LIGHTGBM_model_path

FEATURES_PATH = os.path.join(BASE_DIR, "features.json")
FEATURE_IMPORTANCES_PATH = os.path.join(BASE_DIR, "feature_importances_percentage.json")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
    

with open(FEATURE_IMPORTANCES_PATH) as f:
    feature_importances = json.load(f)
    feature_importances_names = [f["feature"] for f in feature_importances]


with open(FEATURES_PATH) as f:
    feature_info = json.load(f)
    feature_names = [f["name"] for f in feature_info["features"]]
    



def is_model_loaded():
    return 'model' in globals() and model is not None

# curl http://localhost:5000/api/health
@app.route("/api/health", methods=["GET"])
def health():
    try:
        logger.info("Health check request received")
        # Check if model is loaded
        if not is_model_loaded():
            logger.error("Model not loaded during health check")
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
            
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return jsonify({"status": "error", "message": "Model file not found"}), 500
            
        # Check if features file exists
        if not os.path.exists(FEATURES_PATH):
            logger.error("Features file not found")
            return jsonify({"status": "error", "message": "Features file not found"}), 500
            
            
        logger.info("Health check completed successfully")
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_type": type(model).__name__
        })
        
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": f"Health check failed: {str(e)}"}), 500

# curl http://localhost:5000/api/features
@app.route("/api/features", methods=["GET"])
def get_features():
    try:
        logger.info("Features request received")
        if not feature_info:
            logger.error("No features information available")
            return jsonify({"status": "error", "message": "No features information available"}), 500
            
        logger.info(f"Successfully retrieved {len(feature_info['features'])} features")
        return jsonify({
            "status": "success",
            "features": feature_info,
            "count": len(feature_info['features'])
        })
        
    except Exception as e:
        logger.error(f"Failed to retrieve features: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to retrieve features: {str(e)}"
        }), 500
        
        

education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}





@app.route("/api/predict", methods=["POST"])
def predict():
    logger.info("Prediction request received")
    try:
        data = request.get_json()

        # Vérifie que 'education' est présente
        if "education" not in data:
            return jsonify({"error": "Missing 'education' field"}), 400

        # Ajoute automatiquement education.num
        if data["education"] not in education_map:
            return jsonify({"error": f"Unknown education level: {data['education']}"}), 400
        data["education.num"] = education_map[data["education"]]
        
        column_map= {
            "age": "age",
            "workclass": "workclass",
            "education": "education",
            "education.num": "education.num",
            "marital_status": "marital.status",
            "occupation": "occupation",
            "relationship": "relationship",
            "race": "race",
            "sex": "sex",
            "capital_gain": "capital.gain",
            "capital_loss": "capital.loss",
            "hours_per_week": "hours.per.week",
            "native_country": "native.country",
        }
        
        data = {column_map.get(k, k): v for k, v in data.items()}

        print(data)

        # Construction DataFrame + features custom
        df = pd.DataFrame([data])
        df = add_custom_features(df)

        # Prédiction
        proba = model.predict_proba(df)[0]
        y_proba = proba[1]
        prediction = int(y_proba > 0.60)
        label = ">50K" if prediction else "<=50K"

        logger.info(f"Prediction completed: {label} (proba: {y_proba:.4f})")
        return jsonify({
            "prediction": label,
            "probability": round(y_proba, 4),
            "probabilities": dict(zip(model.classes_.astype(str), proba.tolist()))
        })
    
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    try:
        logger.info("Model info request received")
        # Check if model is loaded
        if not is_model_loaded():
            logger.error("Model not loaded during model info request")
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        # Get model attributes safely
        model_attributes = {
            "model_type": type(model).__name__,
            "training_date": "2025-05-23",
            "n_features": len(feature_names) if feature_names else 0,
            "n_estimators": getattr(model, "n_estimators", None),
            "max_depth": getattr(model, "max_depth", None)
        }

        # Validate required fields
        if not model_attributes["model_type"]:
            logger.error("Model type not available")
            raise ValueError("Model type not available")

        logger.info("Model info retrieved successfully")
        return jsonify({
            "status": "success",
            "data": model_attributes
        })

    except Exception as e:
        logger.error(f"Failed to retrieve model information: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to retrieve model information: {str(e)}"
        }), 500


@app.route("/api/explain", methods=["GET"])
def explain():
    try:
        logger.info("Model explanation request received")

        if not model:
            logger.error("Model not loaded during explanation request")
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        # N = int(request.args.get("top", 20))
        # top_features = sorted(feature_importances, key=lambda f: f["percentage"], reverse=True)[:N]

        logger.info("Model explanation completed successfully")
        return jsonify({
            "status": "success",
            "data": feature_importances
        })

    except Exception as e:
        logger.error(f"Failed to explain model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to explain model: {str(e)}"
        }), 500


@app.route("/api/metrics", methods=["GET"])
def metrics():
    METRICS_PATH = os.path.join(BASE_DIR, "GBM_metrics.json")

    try:
        logger.info("Metrics request received")
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        logger.info("Metrics retrieved successfully")
        return jsonify(metrics)
    except FileNotFoundError:
        logger.error("Metrics file not found")
        return jsonify({"error": "Metrics file not found"}), 404
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in metrics file")
        return jsonify({"error": "Invalid JSON format in metrics file"}), 500
    except Exception as e:
        logger.error(f"Unexpected error while retrieving metrics: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/api/model_card", methods=["GET"])
def model_card():
    try:
        logger.info("Model card request received")
        with open("modelCard.json", "r") as f:
            card = json.load(f)
        logger.info("Model card retrieved successfully")
        return jsonify(card)
    except FileNotFoundError:
        logger.error("Model card file not found")
        return jsonify({"error": "Model card file not found"}), 404
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in model card file")
        return jsonify({"error": "Invalid JSON format in model card file"}), 500
    except Exception as e:
        logger.error(f"Unexpected error while retrieving model card: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



