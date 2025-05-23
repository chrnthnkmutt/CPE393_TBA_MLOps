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
    <section className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 py-12">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold mb-4">Income Prediction with Machine Learning</h2>
          <p className="text-xl text-muted-foreground mb-6">
            Use our artificial intelligence model to predict whether a person earns more or less than $50,000
            per year based on their demographic and professional characteristics.
          </p>

          {modelInfo && (
            <div className="flex justify-center items-center gap-4 flex-wrap">
              <Badge variant="outline" className="px-4 py-2 text-sm">
                <Info className="w-4 h-4 mr-2" />
                Model: {modelInfo.model_type}
              </Badge>

              {modelInfo.accuracy_score && (
                <Badge variant="outline" className="px-4 py-2 text-sm">
                  <Award className="w-4 h-4 mr-2" />
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
