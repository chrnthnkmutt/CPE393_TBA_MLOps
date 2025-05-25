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
// import { ModelInfo } from "./model-info"

// import { CompleteAnalysis } from "./complete-analysis"
// import { SeparateAPIs } from "./separate-apis"


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
    <div className="flex flex-col md:flex-row-reverse container mx-auto">
      <div className="flex flex-col gap-4 my-4">
        <ModelInfo info={modelInfoData} isLoading={modelInfoLoading} error={modelInfoError} />
        <ApiList />
        <ModelMetrics metrics={metricsData} isLoading={metricsLoading} error={metricsError} />
      </div>
      
    
    <div className="container mx-auto p-4">
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Vue d'ensemble</TabsTrigger>
          <TabsTrigger value="model">Modèle</TabsTrigger>
          <TabsTrigger value="metrics">Métriques</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>État du système</CardTitle>
              <CardDescription>Vérification de la santé de l'API et des services</CardDescription>
            </CardHeader>
            <CardContent>
              {healthError ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Erreur</AlertTitle>
                  <AlertDescription>
                    Impossible de se connecter à l'API
                  </AlertDescription>
                </Alert>
              ) : healthData?.status === "ok" ? (
                <div className="flex items-center space-x-2">
                  <Badge variant="default">En ligne</Badge>
                  <span>API opérationnelle</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 animate-spin" />
                  <span>Vérification de l'état...</span>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="model">
          <Card>
            <CardHeader>
              <CardTitle>Informations sur le modèle</CardTitle>
              <CardDescription>Détails sur le modèle utilisé</CardDescription>
            </CardHeader>
            <CardContent>
              {modelInfoError ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Erreur</AlertTitle>
                  <AlertDescription>
                    Impossible de récupérer les informations du modèle
                  </AlertDescription>
                </Alert>
              ) : modelInfoData ? (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Server className="h-4 w-4" />
                    <span>Type de modèle: {modelInfoData.model_type}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Zap className="h-4 w-4" />
                    <span>Nombre d'estimateurs: {modelInfoData.n_estimators}</span>
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 animate-spin" />
                  <span>Chargement des informations...</span>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics">
          <Card>
            <CardHeader>
              <CardTitle>Métriques de performance</CardTitle>
              <CardDescription>Évaluation des performances du modèle</CardDescription>
            </CardHeader>
            <CardContent>
              {metricsError ? (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Erreur</AlertTitle>
                  <AlertDescription>
                    Impossible de récupérer les métriques
                  </AlertDescription>
                </Alert>
              ) : metricsData ? (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <span>Précision: {metricsData.accuracy}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span>F1-Score: {metricsData.f1_score}</span>
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 animate-spin" />
                  <span>Chargement des métriques...</span>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
    </div>

  )
}