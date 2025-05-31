import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import  { Badge } from "@/components/ui/badge";
import { Info, Calendar, BrainCircuit, Database, AlertCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { MdElectricBolt } from "react-icons/md";


interface ModelInfoProps {
    info?: {
        data: {
            model_type: string;
            n_estimators: number;
            n_features: number;
            training_date: string;
            max_depth: number | null;
        };
        status: string;
    };
    isLoading: boolean;
    error: Error | null;
}

export function ModelInfo({ info, isLoading, error }: ModelInfoProps) {
    const formattedDate = info?.data?.training_date
      ? new Date(info.data.training_date).toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })
      : null
  
    if (error) {
        return (
            <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-600">
                        <AlertCircle className="w-5 h-5" />
                        Error loading model information
                    </CardTitle>
                    <CardDescription className="text-orange-700">{error.message}</CardDescription>
                </CardHeader>
            </Card>
        )
    }
  
    if (isLoading || !info) {
      return (
        <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-orange-900">
              <Info className="w-5 h-5 text-orange-600" />
              Model Information
            </CardTitle>
            <CardDescription className="text-orange-700">Details about the deployed machine learning model</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2 text-orange-800">
              <Database className="w-3 h-3 mr-1 text-orange-600" />
                Type
              <Skeleton className="h-4 w-[100px] bg-orange-200" />
            </div>
            <div className="flex items-center gap-2 text-orange-800">
              <Calendar className="w-3 h-3 mr-1 text-orange-600" />
                Training Date
              <Skeleton className="h-4 w-[150px] bg-orange-200" />
            </div>
            <div className="flex items-center gap-2 text-orange-800">
              <MdElectricBolt className="w-3 h-3 mr-1 text-orange-600" />
                Number of Estimators
              <Skeleton className="h-4 w-[50px] bg-orange-200" />
            </div>
            <div className="flex items-center gap-2 text-orange-800">
              <BrainCircuit className="w-3 h-3 mr-1 text-orange-600" />
                Number of Features
              <Skeleton className="h-4 w-[50px] bg-orange-200" />
            </div>
            <div className="flex items-center gap-2 text-orange-800">
              <Database className="w-3 h-3 mr-1 text-orange-600" />
                Max Depth
              <Skeleton className="h-4 w-[50px] bg-orange-200" />
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
            Model Information
          </CardTitle>
          <CardDescription className="text-orange-700">Details about the deployed machine learning model</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2 text-orange-800">
              <Database className="w-3 h-3 mr-1 text-orange-600" />
              Type
            <span className="font-medium text-orange-900">{info.data.model_type}</span>
          </div>
  
          {formattedDate && (
            <div className="flex items-center gap-2 text-orange-800">
                <Calendar className="w-3 h-3 mr-1 text-orange-600" />
                Training Date
              <span className="text-orange-900">{formattedDate}</span>
            </div>
          )}
  
          <div className="flex items-center gap-2 text-orange-800">
            <MdElectricBolt className="w-3 h-3 mr-1 text-orange-600" />
              Number of Estimators
            <span className="text-orange-900">{info.data.n_estimators}</span>
          </div>

          <div className="flex items-center gap-2 text-orange-800">
            <BrainCircuit className="w-3 h-3 mr-1 text-orange-600" />
              Number of Features
            <span className="text-orange-900">{info.data.n_features}</span>
          </div>

          {info.data.max_depth && (
            <div className="flex items-center gap-2 text-orange-800">
              <Badge variant="outline" className="px-2 py-1 border-orange-300 text-orange-800 bg-orange-100">
                Max Depth
              </Badge>
              <span className="text-orange-900">{info.data.max_depth}</span>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }