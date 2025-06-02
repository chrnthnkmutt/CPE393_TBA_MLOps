"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart3, TrendingUp } from "lucide-react"

interface FeatureImportanceData {
  data: { [key: string]: number }
  status: string
}

interface FeatureImportanceChartProps {
  data: FeatureImportanceData
  title?: string
  showTop?: number
}

export function FeatureImportanceChart({
  data,
  title = "Feature Importance",
  showTop = 20,
}: FeatureImportanceChartProps) {
  // Vérifier si data.data existe
  if (!data?.data) {
    return (
      <Card className="w-full border-orange-200">
        <CardHeader className="bg-gradient-to-r from-orange-50 to-amber-50">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-6 w-6 text-orange-600" />
            <CardTitle className="text-orange-900">{title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <div className="text-orange-700">No data available</div>
        </CardContent>
      </Card>
    )
  }

  // Trier les features par importance décroissante
  const sortedFeatures = Object.entries(data.data)
    .sort(([, a], [, b]) => b - a)
    .slice(0, showTop)

  // Trouver la valeur maximale pour normaliser les barres
  const maxImportance = Math.max(...Object.values(data.data))

  // Le reste du code reste inchangé...
  const formatFeatureName = (name: string): string => {
    return name
      .replace(/_/g, " ")
      .replace(/\./g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ")
  }

  const getImportanceColor = (importance: number): string => {
    const percentage = (importance / maxImportance) * 100
    if (percentage >= 80) return "bg-orange-600"
    if (percentage >= 60) return "bg-orange-500"
    if (percentage >= 40) return "bg-orange-400"
    if (percentage >= 20) return "bg-orange-300"
    return "bg-orange-200"
  }

  const getFeatureCategory = (name: string): string => {
    if (name.startsWith("education_")) return "Education"
    if (name.startsWith("occupation_")) return "Occupation"
    if (name.startsWith("marital.status_")) return "Marital Status"
    if (name.startsWith("relationship_")) return "Relationship"
    if (name.startsWith("workclass_")) return "Work Class"
    if (name.startsWith("race_")) return "Race"
    if (name.startsWith("sex_")) return "Gender"
    return "Numeric"
  }

  const getCategoryColor = (category: string): string => {
    const colors: { [key: string]: string } = {
      Education: "bg-orange-100 text-orange-800",
      Occupation: "bg-amber-100 text-amber-800",
      "Marital Status": "bg-yellow-100 text-yellow-800",
      Relationship: "bg-orange-100 text-orange-700",
      "Work Class": "bg-red-100 text-red-800",
      Race: "bg-pink-100 text-pink-800",
      Gender: "bg-purple-100 text-purple-800",
      Numeric: "bg-gray-100 text-gray-800",
    }
    return colors[category] || "bg-gray-100 text-gray-800"
  }

  return (
    <Card className="w-full border-orange-200">
      <CardHeader className="bg-gradient-to-r from-orange-50 to-amber-50">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-6 w-6 text-orange-600" />
          <CardTitle className="text-orange-900">{title}</CardTitle>
        </div>
        <CardDescription className="text-orange-700">
          Top {showTop} features sorted by decreasing importance
        </CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-orange-600" />
                <span className="text-sm font-medium text-orange-900">Most important feature</span>
              </div>
              <div className="mt-1">
                <div className="font-semibold text-orange-800">{formatFeatureName(sortedFeatures[0][0])}</div>
                <div className="text-sm text-orange-600">{(sortedFeatures[0][1] * 100).toFixed(2)}%</div>
              </div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
              <div className="text-sm font-medium text-orange-900">Total features</div>
              <div className="text-2xl font-bold text-orange-800">{Object.keys(data.data).length}</div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
              <div className="text-sm font-medium text-orange-900">Cumulative importance (top 10)</div>
              <div className="text-2xl font-bold text-orange-800">
                {(sortedFeatures.slice(0, 10).reduce((sum, [, importance]) => sum + importance, 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="space-y-3">
            {sortedFeatures.map(([feature, importance], index) => {
              const percentage = (importance / maxImportance) * 100
              const category = getFeatureCategory(feature)

              return (
                <div key={feature} className="group hover:bg-orange-50 p-3 rounded-lg transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-8 h-8 bg-orange-100 text-orange-800 rounded-full text-sm font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">{formatFeatureName(feature)}</div>
                        <Badge variant="secondary" className={getCategoryColor(category)}>
                          {category}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-sm font-semibold text-orange-800">
                        {(importance * 100).toFixed(3)}%
                      </div>
                      <div className="text-xs text-gray-500">{percentage.toFixed(1)}% of max</div>
                    </div>
                  </div>

                  <div className="relative">
                    <div className="w-full bg-orange-100 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full transition-all duration-500 ${getImportanceColor(importance)}`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-xs font-medium text-white drop-shadow-sm">
                        {percentage > 15 ? `${percentage.toFixed(1)}%` : ""}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {Object.keys(data.data).length > showTop && (
            <div className="mt-6 p-4 bg-orange-50 border border-orange-200 rounded-lg">
              <p className="text-sm text-orange-700">
                <strong>{Object.keys(data.data).length - showTop}</strong> additional features with lower importance are not displayed.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
