import { Badge } from "@/components/ui/badge"
import { ExternalLink } from "lucide-react"
import { FaGithub } from "react-icons/fa";
import Link from "next/link";

export function Footer() {
  return (
    <footer 
      className="border-t border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50 backdrop-blur supports-[backdrop-filter]:bg-orange-50/90"
      role="contentinfo"
      aria-label="Footer"
    >
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="font-semibold mb-3 text-orange-900" id="about-heading">About the project</h3>
            <p className="text-sm text-orange-700 mb-3" aria-labelledby="about-heading">
              Web interface for demonstrating a machine learning income prediction model. Developed with
              Next.js and Flask for an MLOps project.
            </p>
            <div className="flex gap-2" role="list" aria-label="Technologies">
              <Badge variant="outline" className="text-xs border-orange-300 text-orange-800 bg-orange-100" role="listitem">
                Next.js
              </Badge>
              <Badge variant="outline" className="text-xs border-orange-300 text-orange-800 bg-orange-100" role="listitem">
                Flask
              </Badge>
              <Badge variant="outline" className="text-xs border-orange-300 text-orange-800 bg-orange-100" role="listitem">
                scikit-learn
              </Badge>
            </div>
          </div>

          {/* Technologies */}
          <div>
            <h3 className="font-semibold mb-3 text-orange-900" id="tech-heading">Technologies used</h3>
            <ul 
              className="text-sm text-orange-700 space-y-1"
              aria-labelledby="tech-heading"
            >
              <li>• Frontend: Next.js 15 + TypeScript</li>
              <li>• UI: shadcn/ui + Tailwind CSS</li>
              <li>• Backend: Flask + Python</li>
              <li>• ML: scikit-learn 1.6.0</li>
              <li>• Visualization: Recharts</li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold mb-3 text-orange-900" id="resources-heading">Resources</h3>
            <div className="space-y-2" aria-labelledby="resources-heading">
              <Link
                href="https://github.com/chrnthnkmutt/CPE393_TBA_MLOps"
                className="flex items-center gap-2 text-sm text-orange-700 hover:text-orange-900 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View source code on GitHub"
              >
                <FaGithub className="w-4 h-4" aria-hidden="true" />
                Source code
              </Link>

              <Link
                href="https://github.com/chrnthnkmutt/CPE393_TBA_MLOps"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-orange-700 hover:text-orange-900 transition-colors"
                aria-label="View API documentation"
              >
                <ExternalLink className="w-4 h-4" aria-hidden="true" />
                API Documentation
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t border-orange-200 mt-8 pt-6 text-center">
          <p className="text-sm text-orange-700">
            © 2025 MLOps Income Prediction Interface. Educational demonstration project.
          </p>
        </div>
      </div>
    </footer>
  )
}
