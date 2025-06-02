import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Server } from "lucide-react";

export default function ApiList() {
    return ( 
        <Card className="m-4 md:m-0 border-orange-200 bg-gradient-to-br from-orange-50 to-amber-50">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-orange-900">
                  <Server className="w-5 h-5 text-orange-600" />
                  Endpoints API
                </CardTitle>
            </CardHeader>
            <CardContent>
            <ul className="space-y-3">
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-orange-300 text-orange-800 bg-orange-100">
                    POST
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/predict</span>
                </li>
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-orange-300 text-orange-800 bg-orange-100">
                    POST
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/predict_proba</span>
                </li>
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-orange-300 text-orange-800 bg-orange-100">
                    POST
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/explain</span>
                </li>
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-green-300 text-green-800 bg-green-100">
                    GET
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/features</span>
                </li>
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-green-300 text-green-800 bg-green-100">
                    GET
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/model_info</span>
                </li>
                <li className="flex items-center">
                    <Badge variant="outline" className="mr-2 border-green-300 text-green-800 bg-green-100">
                    GET
                    </Badge>
                    <span className="font-mono text-sm text-orange-900">/api/health</span>
                </li>
            </ul>
            </CardContent>
        </Card>
    )
}