"use client"

import { FeatureImportanceChartLR } from "@/components/home/feature-importance-chart-LR"
import useSWR from 'swr';

const API_BASE_URL = "http://localhost:5000";

const API_ENDPOINTS = {
  features: `${API_BASE_URL}/api/features`, //get features list
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());


const sampleData = {
  data: {
    "age": 0.06320917999236964,
    "capital.gain": 0.0,
    "capital.loss": 0.0,
    "education_Assoc-acdm": 0.1597737770529569,
    "education_Assoc-voc": 0.19128174958407154,
    "education_Bachelors": 0.7462951597765248,
    "education_Doctorate": 2.0134786742912394,
    "education_HS-grad": 0.4500389007750243,
    "education_Masters": 0.7863731050393168,
    "education_Prof-school": 1.5809944103532312,
    "hours.per.week": 0.36698867089908305,
    "marital.status_Married": 0.15877873348005986,
    "marital.status_Never-married": 1.3715492475465174,
    "marital.status_Separated": 0.3690434340777071,
    "marital.status_Widowed": 0.04814700454582399,
    "occupation_Adm-clerical": 0.6235480910553595,
    "occupation_Armed-Forces": 0.04955807356982138,
    "occupation_Craft-repair": 0.5913188816072259,
    "occupation_Exec-managerial": 1.5945251215328997,
    "occupation_Farming-fishing": 0.40237918097773856,
    "occupation_Handlers-cleaners": 0.07060778732293722,
    "occupation_Machine-op-inspct": 0.25492295965332495,
    "occupation_Priv-house-serv": 0.35067384642876376,
    "occupation_Prof-specialty": 1.1249486651419687,
    "occupation_Protective-serv": 1.0934215522618733,
    "occupation_Sales": 0.8300755818351907,
    "occupation_Tech-support": 1.2907093240873515,
    "occupation_Transport-moving": 0.4317769300603967,
    "race_Amer-Indian-Eskimo": 0.47085595605458225,
    "race_Asian-Pac-Islander": 0.09637343991687164,
    "race_Other": 1.3112574804801014,
    "race_White": 0.18091819753956256,
    "relationship_Husband": 0.6503066114975814,
    "relationship_Not-in-family": 0.37071525939117933,
    "relationship_Other-relative": 1.0367735626172885,
    "relationship_Own-child": 1.6774806817516916,
    "relationship_Unmarried": 0.7738804636791998,
    "relationship_Wife": 1.6748764123430155,
    "sex_Female": 0.747338147407988,
    "workclass_Govt_employees": 0.30947828342776457,
    "workclass_Never-worked": 0.021893723006955586,
    "workclass_Private": 0.5845724356550565,
    "workclass_Self_employed": 0.6082024980955162,
    "workclass_Without-pay": 0.009520003413324866
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

        <FeatureImportanceChartLR
          data={sampleData}
        // data={data}
          title="Importance des Features - Modèle de Prédiction de Revenus"
          showTop={15}
        />
      </div>
    </div>
  )
}
