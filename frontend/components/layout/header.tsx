import { Badge } from "@/components/ui/badge"
import { CheckCircle, AlertCircle } from "lucide-react"

interface HeaderProps {
  healthStatus: {
    status: string
  } | null
}

export function Header({ healthStatus }: HeaderProps) {
  return (
    <header 
      className="border-b border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50 backdrop-blur supports-[backdrop-filter]:bg-orange-50/90 sticky top-0 z-50"
      role="banner"
      aria-label="Main header"
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-orange-900" id="main-title">MLOps Interface</h1>
            <p className="text-sm text-orange-700" aria-labelledby="main-title">
              Income Prediction Model
            </p>
          </div>

          {healthStatus && (
            <Badge 
              variant={healthStatus.status === "ok" ? "default" : "destructive"} 
              className={`px-3 py-1 ${
                healthStatus.status === "ok" 
                  ? "bg-green-100 text-green-800 border-green-300 hover:bg-green-200" 
                  : "bg-red-100 text-red-800 border-red-300"
              }`}
              role="status"
              aria-live="polite"
              aria-label={`API status: ${healthStatus.status === "ok" ? "Online" : "Offline"}`}
            >
              {healthStatus.status === "ok" ? (
                <CheckCircle className="w-4 h-4 mr-1" aria-hidden="true" />
              ) : (
                <AlertCircle className="w-4 h-4 mr-1" aria-hidden="true" />
              )}
              API {healthStatus.status === "ok" ? "Online" : "Offline"}
            </Badge>
          )}
        </div>
      </div>
    </header>
  )
}
