import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Info, Target, LineChart, BarChart, TrendingUp, Activity, AlertCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface ModelMetricsProps {
    metrics?: {
        accuracy: number;
        auc: number;
        f1_score: number;
        precision: number;
        recall: number;
    }
    isLoading: boolean;
    error: Error | null;
}

export function ModelMetrics({ metrics, isLoading, error }: ModelMetricsProps) {

    if (error) {
        return (
            <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-600">
                        <AlertCircle className="w-5 h-5" />
                        Error loading model metrics
                    </CardTitle>
                    <CardDescription className="text-orange-700">{error.message}</CardDescription>
                </CardHeader>
            </Card>
        )
    }

    if (isLoading || !metrics) {
        return (
            <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-orange-900">
                        <Info className="w-5 h-5 text-orange-600" />
                        Model Metrics
                    </CardTitle>
                    <CardDescription className="text-orange-700">Performance metrics of the deployed model</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex items-center gap-2 text-orange-800">
                        <Target className="w-3 h-3 mr-1 text-orange-600" />
                        Accuracy
                        <Skeleton className="h-4 w-[100px] bg-orange-200" />
                    </div>
                    <div className="flex items-center gap-2 text-orange-800">
                        <LineChart className="w-3 h-3 mr-1 text-orange-600" />
                        AUC
                        <Skeleton className="h-4 w-[100px] bg-orange-200" />
                    </div>
                    <div className="flex items-center gap-2 text-orange-800">
                        <BarChart className="w-3 h-3 mr-1 text-orange-600" />
                        F1 Score
                        <Skeleton className="h-4 w-[100px] bg-orange-200" />
                    </div>
                    <div className="flex items-center gap-2 text-orange-800">
                        <TrendingUp className="w-3 h-3 mr-1 text-orange-600" />
                        Precision
                        <Skeleton className="h-4 w-[100px] bg-orange-200" />
                    </div>
                    <div className="flex items-center gap-2 text-orange-800">
                        <Activity className="w-3 h-3 mr-1 text-orange-600" />
                        Recall
                        <Skeleton className="h-4 w-[100px] bg-orange-200" />
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-orange-900">
                    <Info className="w-5 h-5 text-orange-600" />
                    Model Metrics
                </CardTitle>
                <CardDescription className="text-orange-700">Performance metrics of the deployed model</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-center gap-2 text-orange-800">
                    <Target className="w-3 h-3 mr-1 text-orange-600" />
                    Accuracy
                    <span className="font-medium text-orange-900">{(metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2 text-orange-800">
                    <LineChart className="w-3 h-3 mr-1 text-orange-600" />
                    AUC
                    <span className="font-medium text-orange-900">{(metrics.auc * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2 text-orange-800">
                    <BarChart className="w-3 h-3 mr-1 text-orange-600" />
                    F1 Score
                    <span className="font-medium text-orange-900">{(metrics.f1_score * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2 text-orange-800">
                    <TrendingUp className="w-3 h-3 mr-1 text-orange-600" />
                    Precision
                    <span className="font-medium text-orange-900">{(metrics.precision * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2 text-orange-800">
                    <Activity className="w-3 h-3 mr-1 text-orange-600" />
                    Recall
                    <span className="font-medium text-orange-900">{(metrics.recall * 100).toFixed(2)}%</span>
                </div>
            </CardContent>
        </Card>
    )
}