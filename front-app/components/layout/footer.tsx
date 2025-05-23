import { Badge } from "@/components/ui/badge"
import { ExternalLink } from "lucide-react"
import { FaGithub } from "react-icons/fa";
import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="font-semibold mb-3">About the project</h3>
            <p className="text-sm text-muted-foreground mb-3">
              Web interface for demonstrating a machine learning income prediction model. Developed with
              Next.js and Flask for an MLOps project.
            </p>
            <div className="flex gap-2">
              <Badge variant="outline" className="text-xs">
                Next.js
              </Badge>
              <Badge variant="outline" className="text-xs">
                Flask
              </Badge>
              <Badge variant="outline" className="text-xs">
                scikit-learn
              </Badge>
            </div>
          </div>

          {/* Technologies */}
          <div>
            <h3 className="font-semibold mb-3">Technologies used</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Frontend: Next.js 15 + TypeScript</li>
              <li>• UI: shadcn/ui + Tailwind CSS</li>
              <li>• Backend: Flask + Python</li>
              <li>• ML: scikit-learn 1.6.0</li>
              <li>• Visualization: Recharts</li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold mb-3">Resources</h3>
            <div className="space-y-2">
              <Link
                href="https://github.com/chrnthnkmutt/CPE393_TBA_MLOps"
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <FaGithub className="w-4 h-4" />
                Source code
              </Link>

              {/* next we need to add the github readme back to the link we will merge web server branch */}
              <Link
                href="https://github.com/chrnthnkmutt/CPE393_TBA_MLOps"
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                API Documentation
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t mt-8 pt-6 text-center">
          <p className="text-sm text-muted-foreground">
            © 2025 MLOps Income Prediction Interface. Educational demonstration project.
          </p>
        </div>
      </div>
    </footer>
  )
}
