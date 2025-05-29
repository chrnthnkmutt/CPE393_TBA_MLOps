from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_model_path = os.path.join(BASE_DIR, "production_models", "RF_model_prod.pkl")
# GBM_model_path = os.path.join(BASE_DIR, "production_models", "GBM_model_prod.pkl")

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
    try:
        logger.info("Health check request received")
        # Check if model is loaded
        if not hasattr(app, 'model'):
            logger.error("Model not loaded during health check")
            return jsonify({"status": "error", "message": "Model not loaded"}), 500
            
        # Check if model file exists
        if not os.path.exists(RF_model_path):
            logger.error(f"Model file not found at {RF_model_path}")
            return jsonify({"status": "error", "message": "Model file not found"}), 500
            
        # Check if features file exists
        if not os.path.exists("features.json"):
            logger.error("Features file not found")
            return jsonify({"status": "error", "message": "Features file not found"}), 500
            
        # Check if model can make a basic prediction
        test_input = {name: 0 for name in feature_names}
        df = pd.DataFrame([test_input])
        try:
            model.predict(df)
        except Exception as e:
            logger.error(f"Model prediction failed during health check: {str(e)}")
            return jsonify({"status": "error", "message": f"Model prediction failed: {str(e)}"}), 500
            
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
            
        logger.info(f"Successfully retrieved {len(feature_info)} features")
        return jsonify({
            "status": "success",
            "features": feature_info,
            "count": len(feature_info)
        })
        
    except Exception as e:
        logger.error(f"Failed to retrieve features: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Failed to retrieve features: {str(e)}"
        }), 500
        
        

# Mapping front → back 
value_map = {
    # éducation
    "education_assoc_acdm":  "education_Assoc-acdm",
    "education_assoc_voc":   "education_Assoc-voc",
    "education_bachelors":   "education_Bachelors",
    "education_doctorate":   "education_Doctorate",
    "education_hs_grad":     "education_HS-grad",
    "education_masters":     "education_Masters",
    "education_prof_school": "education_Prof-school",
    # marital.status
    "marital_status_married":       "marital.status_Married",
    "marital_status_never_married": "marital.status_Never-married",
    "marital_status_separated":     "marital.status_Separated",
    "marital_status_widowed":       "marital.status_Widowed",
    # occupation
    "occupation_adm_clerical":      "occupation_Adm-clerical",
    "occupation_armed_forces":      "occupation_Armed-Forces",
    "occupation_craft_repair":      "occupation_Craft-repair",
    "occupation_exec_managerial":   "occupation_Exec-managerial",
    "occupation_farming_fishing":   "occupation_Farming-fishing",
    "occupation_handlers_cleaners": "occupation_Handlers-cleaners",
    "occupation_machine_op_inspct": "occupation_Machine-op-inspct",
    "occupation_priv_house_serv":   "occupation_Priv-house-serv",
    "occupation_prof_specialty":    "occupation_Prof-specialty",
    "occupation_protective_serv":   "occupation_Protective-serv",
    "occupation_sales":             "occupation_Sales",
    "occupation_tech_support":      "occupation_Tech-support",
    "occupation_transport_moving":  "occupation_Transport-moving",
    # race
    "race_amer_indian_eskimo":      "race_Amer-Indian-Eskimo",
    "race_asian_pac_islander":      "race_Asian-Pac-Islander",
    "race_other":                   "race_Other",
    "race_white":                   "race_White",
    # relationship
    "relationship_husband":         "relationship_Husband",
    "relationship_not_in_family":   "relationship_Not-in-family",
    "relationship_other_relative":  "relationship_Other-relative",
    "relationship_own_child":       "relationship_Own-child",
    "relationship_unmarried":       "relationship_Unmarried",
    "relationship_wife":            "relationship_Wife",
    # workclass
    "workclass_govt_employees":     "workclass_Govt_employees",
    "workclass_never_worked":       "workclass_Never-worked",
    "workclass_private":            "workclass_Private",
    "workclass_self_employed":      "workclass_Self_employed",
    "workclass_without_pay":        "workclass_Without-pay",
    # sex
    "sex_female":                   "sex_Female",
    # NOTE: sex_male → toutes les colonnes `sex_…` restent à 0
}

def transform_payload_to_vector(data: dict) -> dict:

    vec = {feat: 0 for feat in feature_names}

    vec["age"]             = float(data.get("age", 0))
    vec["capital.gain"]    = float(data.get("capital_gain", 0))
    vec["capital.loss"]    = float(data.get("capital_loss", 0))
    vec["hours.per.week"]  = float(data.get("hours_per_week", 0))

    for field in ("education_level", "marital_status", "occupation",
                  "race", "relationship", "workclass", "sex"):
        val = data.get(field)
        if val:
            mapped = value_map.get(val)
            if mapped and mapped in vec:
                vec[mapped] = 1

    return vec



@app.route("/api/predict", methods=["POST"])
def predict():
    logger.info("Prediction request received")
    data = request.get_json()

    # # Check for missing features
    # missing = [name for name in feature_names if name not in data]
    # if missing:
    #     logger.warning(f"Missing features in prediction request: {missing}")

    # Create a vector with missing values replaced by 0
    # input_vector = {name: data.get(name, 0) for name in feature_names}
    input_vector = transform_payload_to_vector(data)
    df = pd.DataFrame([input_vector])

    prediction = model.predict(df)[0]
    label = ">50K" if prediction else "<=50K"
    logger.info(f"Prediction completed: {label}")
    # return jsonify({"prediction": label, "missing_features": missing})
    return jsonify({"prediction": label})


@app.route("/api/predict_proba", methods=["POST"])
def predict_proba():
    logger.info("Probability prediction request received")
    data = request.get_json()
    # missing = [name for name in feature_names if name not in data]

    # if missing:
    #     logger.warning(f"Missing features in probability prediction request: {missing}")

    # input_vector = {name: data.get(name, 0) for name in feature_names}
    input_vector = transform_payload_to_vector(data)
    df = pd.DataFrame([input_vector])
    proba = model.predict_proba(df)[0]

    logger.info("Probability prediction completed successfully")
    return jsonify({
        "probabilities": dict(zip(model.classes_.astype(str), proba.tolist())),
        # "missing_features": missing
    })


def is_model_loaded():
    return 'model' in globals() and model is not None


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



@app.route("/api/explain", methods=["POST"])
def explain():
    try:
        logger.info("Model explanation request received")
        # Check if model is loaded
        if not model:
            logger.error("Model not loaded during explanation request")
            return jsonify({
                "status": "error",
                "message": "Model not loaded"
            }), 500

        # Check if model supports feature importances
        if not hasattr(model, "feature_importances_"):
            logger.error("Model does not provide feature importances")
            return jsonify({
                "status": "error",
                "message": "This model does not provide feature importances"
            }), 400

        # Get feature importances
        importances = dict(zip(feature_names, model.feature_importances_))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        logger.info("Model explanation completed successfully")
        return jsonify({
            "status": "success",
            "data": sorted_importances
        })

    except Exception as e:
        logger.error(f"Failed to explain model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to explain model: {str(e)}"
        }), 500

@app.route("/api/metrics", methods=["GET"])
def metrics():
    try:
        logger.info("Metrics request received")
        with open('RF_metrics.json', 'r') as f:
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
