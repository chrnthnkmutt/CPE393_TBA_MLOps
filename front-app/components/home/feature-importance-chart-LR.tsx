"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart3, TrendingUp, TrendingDown } from "lucide-react"

interface LogisticRegressionCoefData {
  data: { [key: string]: number }
  status: string
}

interface FeatureImportanceChartProps {
  data: LogisticRegressionCoefData
  title?: string
  showTop?: number
}

export function FeatureImportanceChartLR({
  data,
  title = "Logistic Regression Coefficients",
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

  // Trier les features par valeur absolue des coefficients décroissante
  const sortedFeatures = Object.entries(data.data)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, showTop)

  // Trouver la valeur absolue maximale pour normaliser les barres
  const maxAbsCoef = Math.max(...Object.values(data.data).map(Math.abs))

  // Séparer les coefficients positifs et négatifs
  const positiveCoefs = Object.entries(data.data).filter(([, coef]) => coef > 0)
  const negativeCoefs = Object.entries(data.data).filter(([, coef]) => coef < 0)

  const formatFeatureName = (name: string): string => {
    return name
      .replace(/_/g, " ")
      .replace(/\./g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ")
  }

  const getCoefColor = (coefficient: number): string => {
    const absCoef = Math.abs(coefficient)
    const percentage = (absCoef / maxAbsCoef) * 100
    
    if (coefficient > 0) {
      // Couleurs vertes pour les coefficients positifs (augmentent la probabilité)
      if (percentage >= 80) return "bg-green-600"
      if (percentage >= 60) return "bg-green-500"
      if (percentage >= 40) return "bg-green-400"
      if (percentage >= 20) return "bg-green-300"
      return "bg-green-200"
    } else {
      // Couleurs rouges pour les coefficients négatifs (diminuent la probabilité)
      if (percentage >= 80) return "bg-red-600"
      if (percentage >= 60) return "bg-red-500"
      if (percentage >= 40) return "bg-red-400"
      if (percentage >= 20) return "bg-red-300"
      return "bg-red-200"
    }
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

  // Trouver les coefficients les plus extrêmes
  const mostPositive = Object.entries(data.data).reduce((max, [name, coef]) => 
    coef > max[1] ? [name, coef] : max, ["", -Infinity])
  const mostNegative = Object.entries(data.data).reduce((min, [name, coef]) => 
    coef < min[1] ? [name, coef] : min, ["", Infinity])

  return (
    <Card className="w-full border-orange-200">
      <CardHeader className="bg-gradient-to-r from-orange-50 to-amber-50">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-6 w-6 text-orange-600" />
          <CardTitle className="text-orange-900">{title}</CardTitle>
        </div>
        <CardDescription className="text-orange-700">
          Top {showTop} features by absolute coefficient value. Positive coefficients increase income probability, negative decrease it.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium text-green-900">Highest positive</span>
              </div>
              <div className="mt-1">
                <div className="font-semibold text-green-800 text-xs">{formatFeatureName(mostPositive[0])}</div>
                <div className="text-sm text-green-600">+{mostPositive[1].toFixed(3)}</div>
              </div>
            </div>

            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
              <div className="flex items-center space-x-2">
                <TrendingDown className="h-4 w-4 text-red-600" />
                <span className="text-sm font-medium text-red-900">Lowest negative</span>
              </div>
              <div className="mt-1">
                <div className="font-semibold text-red-800 text-xs">{formatFeatureName(mostNegative[0])}</div>
                <div className="text-sm text-red-600">{mostNegative[1].toFixed(3)}</div>
              </div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
              <div className="text-sm font-medium text-orange-900">Total features</div>
              <div className="text-2xl font-bold text-orange-800">{Object.keys(data.data).length}</div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
              <div className="text-sm font-medium text-orange-900">Positive / Negative</div>
              <div className="text-lg font-bold text-orange-800">
                <span className="text-green-600">{positiveCoefs.length}</span>
                {" / "}
                <span className="text-red-600">{negativeCoefs.length}</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            {sortedFeatures.map(([feature, coefficient], index) => {
              const absCoef = Math.abs(coefficient)
              const percentage = (absCoef / maxAbsCoef) * 100
              const category = getFeatureCategory(feature)
              const isPositive = coefficient > 0

              return (
                <div key={feature} className="group hover:bg-orange-50 p-3 rounded-lg transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center justify-center w-8 h-8 bg-orange-100 text-orange-800 rounded-full text-sm font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">{formatFeatureName(feature)}</div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="secondary" className={getCategoryColor(category)}>
                            {category}
                          </Badge>
                          <Badge variant={isPositive ? "default" : "destructive"} className="text-xs">
                            {isPositive ? "Increases" : "Decreases"} probability
                          </Badge>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-mono text-sm font-semibold ${isPositive ? 'text-green-700' : 'text-red-700'}`}>
                        {isPositive ? '+' : ''}{coefficient.toFixed(4)}
                      </div>
                      <div className="text-xs text-gray-500">|{absCoef.toFixed(4)}| magnitude</div>
                    </div>
                  </div>

                  <div className="relative">
                    <div className="w-full bg-gray-100 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full transition-all duration-500 ${getCoefColor(coefficient)}`}
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
                <strong>{Object.keys(data.data).length - showTop}</strong> additional features with smaller absolute coefficients are not displayed.
              </p>
            </div>
          )}

          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-semibold text-blue-900 mb-2">Understanding Logistic Regression Coefficients:</h4>
            <ul className="text-xs text-blue-700 space-y-1">
              <li>• <strong>Positive coefficients</strong> increase the probability of income &gt; $50K</li>
              <li>• <strong>Negative coefficients</strong> decrease the probability of income &gt; $50K</li>
              <li>• <strong>Magnitude</strong> indicates the strength of the effect</li>
              <li>• Coefficients are in log-odds units</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
