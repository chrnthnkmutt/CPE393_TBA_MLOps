import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import  { Badge } from "@/components/ui/badge";
import { Info, Calendar, BrainCircuit, Database, AlertCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { MdElectricBolt } from "react-icons/md";


interface ModelInfoProps {
    info?: {
        model_type: string;
        n_estimators: number;
        n_features: number;
        training_date: string;
        max_depth: number | null;
        [key: string]: any;
    }
    isLoading: boolean;
    error: Error | null;
}

export function ModelInfo({ info, isLoading, error }: ModelInfoProps) {
    const formattedDate = info?.training_date
      ? new Date(info.training_date).toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })
      : null
  
    if (error) {
        return (
            <Card className="m-4 md:m-0">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-500">
                        <AlertCircle className="w-5 h-5" />
                        Error loading model information
                    </CardTitle>
                    <CardDescription>{error.message}</CardDescription>
                </CardHeader>
            </Card>
        )
    }
  
    if (isLoading || !info) {
      return (
        <Card className="m-4 md:m-0">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="w-5 h-5" />
              Model Information
            </CardTitle>
            <CardDescription>Details about the deployed machine learning model</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2">
              <Database className="w-3 h-3 mr-1" />
                Type
              <Skeleton className="h-4 w-[100px]" />
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="w-3 h-3 mr-1" />
                Training Date
              <Skeleton className="h-4 w-[150px]" />
            </div>
            <div className="flex items-center gap-2">
              <MdElectricBolt className="w-3 h-3 mr-1" />
                Number of Estimators
              <Skeleton className="h-4 w-[50px]" />
            </div>
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-3 h-3 mr-1" />
                Number of Features
              <Skeleton className="h-4 w-[50px]" />
            </div>
            <div className="flex items-center gap-2">
              <Database className="w-3 h-3 mr-1" />
                Max Depth
              <Skeleton className="h-4 w-[50px]" />
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
            Model Information
          </CardTitle>
          <CardDescription>Details about the deployed machine learning model</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2">
              <Database className="w-3 h-3 mr-1" />
              Type
            <span className="font-medium">{info.model_type}</span>
          </div>
  
          {formattedDate && (
            <div className="flex items-center gap-2">
                <Calendar className="w-3 h-3 mr-1" />
                Training Date
              <span>{formattedDate}</span>
            </div>
          )}
  
          <div className="flex items-center gap-2">
            <MdElectricBolt className="w-3 h-3 mr-1" />
              Number of Estimators
            <span>{info.n_estimators}</span>
          </div>

          <div className="flex items-center gap-2">
            <BrainCircuit className="w-3 h-3 mr-1" />
              Number of Features
            
            <span>{info.n_features}</span>
          </div>

          {info.max_depth && (
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="px-2 py-1">
                Max Depth
              </Badge>
              <span>{info.max_depth}</span>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }