// Features data based on the provided list
export const features = [
    {
      name: "education_Assoc-acdm",
      type: "binary",
      group: "education",
    },
    {
      name: "education_Assoc-voc",
      type: "binary",
      group: "education",
    },
    {
      name: "education_Bachelors",
      type: "binary",
      group: "education",
    },
    {
      name: "education_Doctorate",
      type: "binary",
      group: "education",
    },
    {
      name: "education_HS-grad",
      type: "binary",
      group: "education",
    },
    {
      name: "education_Masters",
      type: "binary",
      group: "education",
    },
    {
      name: "education_Prof-school",
      type: "binary",
      group: "education",
    },
    {
      name: "marital.status_Married",
      type: "binary",
      group: "marital.status",
    },
    {
      name: "marital.status_Never-married",
      type: "binary",
      group: "marital.status",
    },
    {
      name: "marital.status_Separated",
      type: "binary",
      group: "marital.status",
    },
    {
      name: "marital.status_Widowed",
      type: "binary",
      group: "marital.status",
    },
    {
      name: "occupation_Adm-clerical",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Armed-Forces",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Craft-repair",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Exec-managerial",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Farming-fishing",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Handlers-cleaners",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Machine-op-inspct",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Priv-house-serv",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Prof-specialty",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Protective-serv",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Sales",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Tech-support",
      type: "binary",
      group: "occupation",
    },
    {
      name: "occupation_Transport-moving",
      type: "binary",
      group: "occupation",
    },
    {
      name: "race_Amer-Indian-Eskimo",
      type: "binary",
      group: "race",
    },
    {
      name: "race_Asian-Pac-Islander",
      type: "binary",
      group: "race",
    },
    {
      name: "race_Other",
      type: "binary",
      group: "race",
    },
    {
      name: "race_White",
      type: "binary",
      group: "race",
    },
    {
      name: "relationship_Husband",
      type: "binary",
      group: "relationship",
    },
    {
      name: "relationship_Not-in-family",
      type: "binary",
      group: "relationship",
    },
    {
      name: "relationship_Other-relative",
      type: "binary",
      group: "relationship",
    },
    {
      name: "relationship_Own-child",
      type: "binary",
      group: "relationship",
    },
    {
      name: "relationship_Unmarried",
      type: "binary",
      group: "relationship",
    },
    {
      name: "relationship_Wife",
      type: "binary",
      group: "relationship",
    },
    {
      name: "sex_Female",
      type: "binary",
      group: "sex",
    },
    {
      name: "workclass_Govt_employees",
      type: "binary",
      group: "workclass",
    },
    {
      name: "workclass_Never-worked",
      type: "binary",
      group: "workclass",
    },
    {
      name: "workclass_Private",
      type: "binary",
      group: "workclass",
    },
    {
      name: "workclass_Self_employed",
      type: "binary",
      group: "workclass",
    },
    {
      name: "workclass_Without-pay",
      type: "binary",
      group: "workclass",
    },
    {
      name: "age",
      type: "numeric",
      min: 17,
      max: 90,
      default: 38,
    },
    {
      name: "capital.gain",
      type: "numeric",
      min: 0,
      max: 99999,
      default: 0,
    },
    {
      name: "capital.loss",
      type: "numeric",
      min: 0,
      max: 4356,
      default: 0,
    },
    {
      name: "hours.per.week",
      type: "numeric",
      min: 1,
      max: 99,
      default: 40,
    },
  ]
  
  // Group features by their category
  export const featureGroups = features.reduce<Record<string, typeof features>>((groups, feature) => {
    if (feature.group) {
      if (!groups[feature.group]) {
        groups[feature.group] = []
      }
      groups[feature.group].push(feature)
    }
    return groups
  }, {})
  
  // Get numeric features
  export const numericFeatures = features.filter((feature) => feature.type === "numeric")
  