"use client"

import { Badge } from "@/components/ui/badge"
import { CheckCircle, AlertCircle } from "lucide-react"
import Link from "next/link"
import useSWR from 'swr'

const fetcher = (url: string) => fetch(url).then((res) => res.json());
const API_BASE_URL = "http://localhost:5000";

export function Header() {
  const { data: healthData, error: healthError } = useSWR(
    `${API_BASE_URL}/api/health`, 
    fetcher,
    { 
      refreshInterval: 30000, // Refresh every 30 seconds
      errorRetryCount: 3,
      dedupingInterval: 10000
    }
  );

  // Determine health status
  const healthStatus = healthError 
    ? { status: "error" } 
    : healthData?.status === "ok" 
      ? { status: "ok" } 
      : null;

  return (
    <header 
      className="border-b border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50 backdrop-blur supports-[backdrop-filter]:bg-orange-50/90 sticky top-0 z-50"
      role="banner"
      aria-label="Main header"
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div>
              <h1 className="text-2xl font-bold text-orange-900" id="main-title">MLOps Interface</h1>
              <p className="text-sm text-orange-700" aria-labelledby="main-title">
                Income Prediction Model
              </p>
            </div>
            
            <nav className="flex items-center gap-6" role="navigation" aria-label="Navigation principale">
              <Link 
                href="/" 
                className="text-orange-700 hover:text-orange-900 font-medium transition-colors duration-200 hover:underline"
                aria-label="Retour au menu principal"
              >
                Menu
              </Link>
              <Link 
                href="/feature-importance" 
                className="text-orange-700 hover:text-orange-900 font-medium transition-colors duration-200 hover:underline"
                aria-label="Explication des fonctionnalités"
              >
                Feature Explanation
              </Link>
            </nav>
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
