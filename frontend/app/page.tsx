import { Hero } from "@/components/home/hero";
import { MainContent } from "@/components/home/main-content";


export default function Home() {
  return (
    <article>
        <Hero modelInfo={null} />
        <MainContent />
    </article>
  );
}
