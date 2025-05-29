"use client"

import { FeatureImportanceChart } from "@/components/home/feature-importance-chart"
import useSWR from 'swr';

const API_BASE_URL = "http://localhost:5000";

const API_ENDPOINTS = {
  features: `${API_BASE_URL}/api/features`, //get features list
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());


const sampleData = {
  data: {
    age: 0.37293559582398894,
    "capital.gain": 0.0,
    "capital.loss": 0.0,
    "education_Assoc-acdm": 0.007881611835858327,
    "education_Assoc-voc": 0.009356314658635727,
    education_Bachelors: 0.02543649591197348,
    education_Doctorate: 0.008871248152555964,
    "education_HS-grad": 0.03432523901657315,
    education_Masters: 0.015050241950287894,
    "education_Prof-school": 0.008740030661183049,
    "hours.per.week": 0.0998995843241709,
    "marital.status_Married": 0.06952392752725614,
    "marital.status_Never-married": 0.038960963756152704,
    "marital.status_Separated": 0.012856919530016697,
    "marital.status_Widowed": 0.0023187649923606455,
    "occupation_Adm-clerical": 0.010056627802482327,
    "occupation_Armed-Forces": 1.4084895559464618e-6,
    "occupation_Craft-repair": 0.009970293198913377,
    "occupation_Exec-managerial": 0.0282403257016011,
    "occupation_Farming-fishing": 0.004072247808932837,
    "occupation_Handlers-cleaners": 0.00470212349197176,
    "occupation_Machine-op-inspct": 0.006530991813425924,
    "occupation_Priv-house-serv": 0.00015205526484627244,
    "occupation_Prof-specialty": 0.020044614195720224,
    "occupation_Protective-serv": 0.005382930345716534,
    occupation_Sales: 0.00996918199088381,
    "occupation_Tech-support": 0.007734134429948438,
    "occupation_Transport-moving": 0.006255302466366095,
    "race_Amer-Indian-Eskimo": 0.00225114582232715,
    "race_Asian-Pac-Islander": 0.007446750605848536,
    race_Other: 0.0022358227529066833,
    race_White: 0.018200296162094867,
    relationship_Husband: 0.04576658542381273,
    "relationship_Not-in-family": 0.017211705563012623,
    "relationship_Other-relative": 0.0024762224070195935,
    "relationship_Own-child": 0.010536792394087424,
    relationship_Unmarried: 0.008361797320329087,
    relationship_Wife: 0.0163452621471333,
    sex_Female: 0.017313900205767927,
    workclass_Govt_employees: 0.011302872173192646,
    "workclass_Never-worked": 0.0,
    workclass_Private: 0.012904568236610221,
    workclass_Self_employed: 0.008377102750028799,
    "workclass_Without-pay": 8.94450124841307e-10,
  },
  status: "success",
}

export default function FeatureImportanceDemo() {
  const { data, error } = useSWR(API_ENDPOINTS.features, fetcher);

  if (error) return <div>Failed to load</div>;
  if (!data) return <div>Loading...</div>;

  console.log(data);
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-50 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-orange-900 mb-2">Feature Importance Analysis</h1>
          <p className="text-orange-700">Visualization of the most important features of your ML model</p>
        </div>

        <FeatureImportanceChart
          data={sampleData}
        // data={data}
          title="Importance des Features - Modèle de Prédiction de Revenus"
          showTop={15}
        />
      </div>
    </div>
  )
}
