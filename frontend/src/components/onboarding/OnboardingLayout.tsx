import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

interface OnboardingLayoutProps {
  step: number;
  totalSteps?: number;
  children: React.ReactNode;
  onBack?: () => void;
  showBack?: boolean;
}

export default function OnboardingLayout({
  step,
  totalSteps = 5,
  children,
  onBack,
  showBack = true,
}: OnboardingLayoutProps) {
  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <div className="absolute inset-0 clypt-grid-bg" />
      <div className="absolute inset-0 clypt-radial-glow" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] rounded-full bg-primary/[0.03] blur-[120px]" />

      {/* Header */}
      <header className="relative z-10 h-16 flex items-center justify-between px-6 md:px-10">
        <Link to="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="font-display font-extrabold text-primary-foreground text-sm">C</span>
          </div>
          <span className="font-display font-bold text-foreground text-lg tracking-tight">Clypt</span>
        </Link>

        {/* Step indicator */}
        <div className="flex items-center gap-2">
          {Array.from({ length: totalSteps }).map((_, i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i < step
                  ? "w-6 bg-primary"
                  : i === step
                  ? "w-6 bg-primary/60"
                  : "w-1.5 bg-border"
              }`}
            />
          ))}
        </div>
      </header>

      {/* Back button */}
      {showBack && onBack && (
        <div className="relative z-10 px-6 md:px-10 mt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="gap-1.5 text-muted-foreground hover:text-foreground text-xs font-mono"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back
          </Button>
        </div>
      )}

      {/* Content */}
      <motion.div
        key={step}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10"
      >
        {children}
      </motion.div>
    </div>
  );
}
