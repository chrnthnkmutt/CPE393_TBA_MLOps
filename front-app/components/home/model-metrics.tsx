import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
            <Card className="m-4 md:m-0">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-500">
                        <AlertCircle className="w-5 h-5" />
                        Error loading model metrics
                    </CardTitle>
                    <CardDescription>{error.message}</CardDescription>
                </CardHeader>
            </Card>
        )
    }

    if (isLoading || !metrics) {
        return (
            <Card className="m-4 md:m-0">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Info className="w-5 h-5" />
                        Model Metrics
                    </CardTitle>
                    <CardDescription>Performance metrics of the deployed model</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex items-center gap-2">
                        <Target className="w-3 h-3 mr-1" />
                        Accuracy
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                    <div className="flex items-center gap-2">
                        <LineChart className="w-3 h-3 mr-1" />
                        AUC
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                    <div className="flex items-center gap-2">
                        <BarChart className="w-3 h-3 mr-1" />
                        F1 Score
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                    <div className="flex items-center gap-2">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        Precision
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                    <div className="flex items-center gap-2">
                        <Activity className="w-3 h-3 mr-1" />
                        Recall
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card className="m-4 md:m-0">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Info className="w-5 h-5" />
                    Model Metrics
                </CardTitle>
                <CardDescription>Performance metrics of the deployed model</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-center gap-2">
                    <Target className="w-3 h-3 mr-1" />
                    Accuracy
                    <span className="font-medium">{(metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2">
                    <LineChart className="w-3 h-3 mr-1" />
                    AUC
                    <span className="font-medium">{(metrics.auc * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2">
                    <BarChart className="w-3 h-3 mr-1" />
                    F1 Score
                    <span className="font-medium">{(metrics.f1_score * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2">
                    <TrendingUp className="w-3 h-3 mr-1" />
                    Precision
                    <span className="font-medium">{(metrics.precision * 100).toFixed(2)}%</span>
                </div>
                <div className="flex items-center gap-2">
                    <Activity className="w-3 h-3 mr-1" />
                    Recall
                    <span className="font-medium">{(metrics.recall * 100).toFixed(2)}%</span>
                </div>
            </CardContent>
        </Card>
    )
}

// {
//     "accuracy": 0.7961099932930918,
//     "auc": 0.6713371739959909,
//     "f1_score": 0.8728033472803347,
//     "precision": 0.8577302631578947,
//     "recall": 0.8884156729131175
// }

