import pickle
import pandas as pd

# Loading the model
with open('RF_model_prod.pkl', 'rb') as f:
    model = pickle.load(f)

# Input data
data = {
    "education_Doctorate": 1,
    "education_Bachelors": 0,
    "education_Assoc-acdm": 0,
    "education_Assoc-voc": 0,
    "education_HS-grad": 0,
    "education_Masters": 0,
    "education_Prof-school": 0,
    "marital.status_Married": 1,
    "marital.status_Never-married": 0,
    "marital.status_Separated": 0,
    "marital.status_Widowed": 0,
    "occupation_Prof-specialty": 1,
    "occupation_Exec-managerial": 0,
    "occupation_Adm-clerical": 0,
    "occupation_Armed-Forces": 0,
    "occupation_Craft-repair": 0,
    "occupation_Farming-fishing": 0,
    "occupation_Handlers-cleaners": 0,
    "occupation_Machine-op-inspct": 0,
    "occupation_Priv-house-serv": 0,
    "occupation_Protective-serv": 0,
    "occupation_Sales": 0,
    "occupation_Tech-support": 0,
    "occupation_Transport-moving": 0,
    "race_White": 1,
    "race_Amer-Indian-Eskimo": 0,
    "race_Asian-Pac-Islander": 0,
    "race_Other": 0,
    "relationship_Husband": 1,
    "relationship_Not-in-family": 0,
    "relationship_Other-relative": 0,
    "relationship_Own-child": 0,
    "relationship_Unmarried": 0,
    "relationship_Wife": 0,
    "sex_Female": 0,
    "workclass_Private": 1,
    "workclass_Govt_employees": 0,
    "workclass_Never-worked": 0,
    "workclass_Self_employed": 0,
    "workclass_Without-pay": 0,
    "age": 52,
    "capital.gain": 99999,
    "capital.loss": 0,
    "hours.per.week": 80
}

print("Columns expected by the model:")
print(model.feature_names_in_)

missing = [feat for feat in model.feature_names_in_ if feat not in data]
print("\nMissing features in your input:")
print(missing)

# Reorder features according to those learned by the model
ordered_data = {name: data.get(name, 0) for name in model.feature_names_in_}

# Create properly ordered DataFrame
df = pd.DataFrame([ordered_data])

# Prediction
print("Raw prediction:", model.predict(df)[0])
print("Probabilities:", model.predict_proba(df)[0])
print(model.classes_)
