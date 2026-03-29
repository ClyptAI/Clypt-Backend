import { useMemo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Rocket, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ItemCarousel } from "@/components/tool-ui/item-carousel";
import OnboardingLayout from "@/components/onboarding/OnboardingLayout";
import { mockTopShorts } from "@/data/mockOnboarding";

const completedItems = [
  "Channel connected & verified",
  "Content library scanned",
  "Creator brand profile generated",
  "Clip preferences configured",
];

function formatDuration(seconds: number): string {
  if (!seconds || seconds <= 0) return "0s";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m === 0) return `${s}s`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function OnboardReady() {
  const navigate = useNavigate();
  const location = useLocation();
  const { creatorId, recentItems } = (location.state as any) || {};

  const carouselItems = useMemo(() => {
    if (recentItems && Array.isArray(recentItems) && recentItems.length > 0) {
      return recentItems.map((item: any, i: number) => ({
        id: item.video_id || `item-${i}`,
        name: item.title || "Untitled",
        subtitle: formatDuration(item.duration_seconds ?? 0),
        color: `hsl(0 ${60 + i * 3}% ${35 + i * 2}%)`,
      }));
    }
    return mockTopShorts.map((short, i) => ({
      id: `short-${i}`,
      name: short.title,
      subtitle: `${short.views} views · ${short.duration}`,
      color: `hsl(0 ${60 + i * 3}% ${35 + i * 2}%)`,
    }));
  }, [recentItems]);

  return (
    <OnboardingLayout step={4} onBack={() => navigate("/onboard/clip-preferences")}>
      <div className="flex flex-col items-center justify-center px-6 pt-16 pb-20 max-w-lg mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-10"
        >
          <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-6 relative">
            <Rocket className="w-8 h-8 text-primary" />
            <div className="absolute -inset-1 rounded-2xl bg-primary/10 blur-md" />
          </div>
          <h1 className="font-display text-3xl font-extrabold text-foreground">You're all set!</h1>
          <p className="text-sm text-muted-foreground mt-3 max-w-sm mx-auto">
            Clypt is ready to analyze your videos and generate clips tailored to your brand.
          </p>
        </motion.div>

        {/* Checklist */}
        <div className="w-full rounded-xl clypt-glass p-6 mb-6">
          <div className="space-y-3">
            {completedItems.map((item, i) => (
              <motion.div
                key={item}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                className="flex items-center gap-3"
              >
                <CheckCircle2 className="w-4.5 h-4.5 text-primary shrink-0" />
                <span className="text-sm text-foreground">{item}</span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Scanned Videos Carousel */}
        <div className="w-full rounded-xl clypt-glass p-5 mb-8 [&_*]:text-foreground">
          <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-1">
            {recentItems ? "Videos analyzed" : "Your Top-Performing Shorts"}
          </p>
          <p className="text-xs text-muted-foreground mb-4">
            {recentItems ? "These videos were used to build your creator profile." : "These are the clips your audience loves most."}
          </p>
          <ItemCarousel
            id="top-shorts-carousel"
            items={carouselItems}
          />
        </div>

        <Button
          onClick={() => navigate(`/run/${creatorId || "demo"}`)}
          size="lg"
          className="w-full max-w-sm h-12 gap-2 font-display font-bold rounded-xl text-sm"
        >
          Go to dashboard
          <ArrowRight className="w-4 h-4" />
        </Button>
      </div>
    </OnboardingLayout>
  );
}
