import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Server } from "lucide-react";

export default function ApiList() {
    return ( 
        <Card className="m-4 md:m-0">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="w-5 h-5" />
                  Endpoints API
                </CardTitle>
            </CardHeader>
            <CardContent>
            <ul className="space-y-3">
                <li>
                    <Badge variant="outline" className="mr-2">
                    POST
                    </Badge>
                    <span className="font-mono text-sm">/api/predict</span>
                </li>
                <li>
                    <Badge variant="outline" className="mr-2">
                    POST
                    </Badge>
                    <span className="font-mono text-sm">/api/predict_proba</span>
                </li>
                <li>
                    <Badge variant="outline" className="mr-2">
                    POST
                    </Badge>
                    <span className="font-mono text-sm">/api/explain</span>
                </li>
                <li>
                    <Badge variant="outline" className="mr-2">
                    GET
                    </Badge>
                    <span className="font-mono text-sm">/api/features</span>
                </li>
                <li>
                    <Badge variant="outline" className="mr-2">
                    GET
                    </Badge>
                    <span className="font-mono text-sm">/api/model_info</span>
                </li>
                <li>
                    <Badge variant="outline" className="mr-2">
                    GET
                    </Badge>
                    <span className="font-mono text-sm">/api/health</span>
                </li>
            </ul>
            </CardContent>
        </Card>
    )
}