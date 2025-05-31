import json

features = [
    'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
    'education_Doctorate', 'education_HS-grad', 'education_Masters',
    'education_Prof-school', 'marital.status_Married',
    'marital.status_Never-married', 'marital.status_Separated',
    'marital.status_Widowed', 'occupation_Adm-clerical',
    'occupation_Armed-Forces', 'occupation_Craft-repair',
    'occupation_Exec-managerial', 'occupation_Farming-fishing',
    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Priv-house-serv', 'occupation_Prof-specialty',
    'occupation_Protective-serv', 'occupation_Sales',
    'occupation_Tech-support', 'occupation_Transport-moving',
    'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Other',
    'race_White', 'relationship_Husband', 'relationship_Not-in-family',
    'relationship_Other-relative', 'relationship_Own-child',
    'relationship_Unmarried', 'relationship_Wife', 'sex_Female',
    'workclass_Govt_employees', 'workclass_Never-worked',
    'workclass_Private', 'workclass_Self_employed', 'workclass_Without-pay',
    'age', 'capital.gain', 'capital.loss', 'hours.per.week'
]

features_json = []

for f in features:
    if f in ['age', 'capital.gain', 'capital.loss', 'hours.per.week']:
        features_json.append({ "name": f, "type": "numeric" })
    else:
        features_json.append({ "name": f, "type": "binary" })

with open("features.json", "w") as f:
    json.dump(features_json, f, indent=2)
