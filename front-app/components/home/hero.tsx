import { Badge } from "@/components/ui/badge"
import { Info, Award } from "lucide-react"

interface HeroProps {
  modelInfo: {
    model_type?: string
    accuracy_score?: number
    [key: string]: any
  } | null
}

export function Hero({ modelInfo }: HeroProps) {
  return (
    <section 
      className="bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-950/20 dark:to-amber-950/20 py-12"
      role="banner"
      aria-label="Hero section"
    >
      <div className="container mx-auto px-4">
        <div className="text-center max-w-4xl mx-auto">
          <h2 
            className="text-4xl font-bold mb-4 text-orange-900"
            id="hero-title"
          >
            Income Prediction with Machine Learning
          </h2>
          <p 
            className="text-xl text-orange-700 mb-6"
            aria-labelledby="hero-title"
          >
            Use our artificial intelligence model to predict whether a person earns more or less than $50,000
            per year based on their demographic and professional characteristics.
          </p>

          {modelInfo && (
            <div 
              className="flex justify-center items-center gap-4 flex-wrap"
              role="list"
              aria-label="Model information"
            >
              <Badge 
                variant="outline" 
                className="px-4 py-2 text-sm border-orange-300 text-orange-800 bg-orange-50"
                role="listitem"
              >
                <Info className="w-4 h-4 mr-2 text-orange-600" aria-hidden="true" />
                Model: {modelInfo.model_type}
              </Badge>

              {modelInfo.accuracy_score && (
                <Badge 
                  variant="outline" 
                  className="px-4 py-2 text-sm border-orange-300 text-orange-800 bg-orange-50"
                  role="listitem"
                >
                  <Award className="w-4 h-4 mr-2 text-orange-600" aria-hidden="true" />
                  Accuracy: {(modelInfo.accuracy_score * 100).toFixed(2)}%
                </Badge>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
