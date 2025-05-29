"use client"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, Activity, Server, Zap } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import useSWR from 'swr';
import ApiList from "./apiList"
import { ModelInfo } from "./model-info"
import { ModelMetrics } from "./model-metrics"
// import { CompleteAnalysis } from "./complete-analysis"
import { CompleteAnalysisOrange } from "./complete-analysisOrange"

/*
api list :

local : http://localhost:5000/

/api/predict_proba
/api/explain
/api/health
/api/model_info
/api/features
/api/metrics
*/

const fetcher = (url: string) => fetch(url).then((res) => res.json());

const API_BASE_URL = "http://localhost:5000";
// const API_BASE_URL = "http://backend:5000";

const API_ENDPOINTS = {
  health: `${API_BASE_URL}/api/health`,
  modelInfo: `${API_BASE_URL}/api/model_info`,
  features: `${API_BASE_URL}/api/features`, //get features list
  predictProba: `${API_BASE_URL}/api/predict_proba`,
  explain: `${API_BASE_URL}/api/explain`,
  metrics: `${API_BASE_URL}/api/metrics`,
}

export function MainContent() {
  const { data: healthData, error: healthError, isLoading: healthLoading } = useSWR(API_ENDPOINTS.health, fetcher);
  const { data: modelInfoData, error: modelInfoError, isLoading: modelInfoLoading } = useSWR(API_ENDPOINTS.modelInfo, fetcher);
  const { data: featuresData, error: featuresError, isLoading: featuresLoading } = useSWR(API_ENDPOINTS.features, fetcher);
  const { data: predictProbaData, error: predictProbaError, isLoading: predictProbaLoading } = useSWR(API_ENDPOINTS.predictProba, fetcher);
  const { data: explainData, error: explainError, isLoading: explainLoading } = useSWR(API_ENDPOINTS.explain, fetcher);
  const { data: metricsData, error: metricsError, isLoading: metricsLoading } = useSWR(API_ENDPOINTS.metrics, fetcher);

  return (
    <div className="flex flex-col md:flex-row-reverse container mx-auto bg-gradient-to-br from-orange-50 to-amber-50 min-h-screen">
      <div className="flex flex-col gap-4 my-4 p-4">
        <div className="space-y-4">
          <ModelInfo info={modelInfoData} isLoading={modelInfoLoading} error={modelInfoError} />
          <ApiList />
          <ModelMetrics metrics={metricsData} isLoading={metricsLoading} error={metricsError} />
        </div>
      </div>
      {/* <CompleteAnalysis /> */}
      <div className="flex-1">
        <CompleteAnalysisOrange />
      </div>
    </div>
  )
}