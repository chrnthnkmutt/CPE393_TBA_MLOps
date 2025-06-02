"use client"

import { FeatureImportanceChartLightGBM } from "@/components/home/feature-importance-chart-lightGBM"
import useSWR from 'swr';

const API_BASE_URL = "http://localhost:5000";

const API_ENDPOINTS = {
  features: `${API_BASE_URL}/api/explain`, //get features list
}

const fetcher = (url: string) => fetch(url).then((res) => res.json());



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

        <FeatureImportanceChartLightGBM
          data={data.data}
          title="Importance des Features - Modèle de Prédiction de Revenus"
          showTop={20}
        />
      </div>
    </div>
  )
}
