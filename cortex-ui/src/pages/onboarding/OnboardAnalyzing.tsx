import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Brain } from "lucide-react";
import OnboardingLayout from "@/components/onboarding/OnboardingLayout";
import { ProgressTracker } from "@/components/tool-ui/progress-tracker";
import type { ProgressStep } from "@/components/tool-ui/progress-tracker";

const initialSteps: ProgressStep[] = [
  { id: "metadata", label: "Fetching channel metadata", description: "Pulling subscriber count, upload frequency, and channel info", status: "pending" },
  { id: "shorts", label: "Scanning top-performing shorts", description: "Analyzing your 10 best-performing short-form clips", status: "pending" },
  { id: "longform", label: "Analyzing long-form content", description: "Reviewing structure and engagement of full videos", status: "pending" },
  { id: "patterns", label: "Extracting brand patterns", description: "Identifying recurring themes, hooks, and audience triggers", status: "pending" },
  { id: "profile", label: "Building your creator profile", description: "Synthesizing everything into your Brand DNA", status: "pending" },
];

export default function OnboardAnalyzing() {
  const navigate = useNavigate();
  const [steps, setSteps] = useState<ProgressStep[]>(initialSteps);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let currentStep = 0;

    // Start first step immediately
    setSteps((prev) => prev.map((s, i) => (i === 0 ? { ...s, status: "in-progress" as const } : s)));

    const timer = setInterval(() => {
      setElapsed((e) => e + 100);

      setSteps((prev) => {
        if (currentStep >= prev.length) return prev;

        // Complete current step and start next
        currentStep++;
        return prev.map((s, i) => {
          if (i < currentStep) return { ...s, status: "completed" as const };
          if (i === currentStep) return { ...s, status: "in-progress" as const };
          return s;
        });
      });

      if (currentStep >= initialSteps.length) {
        clearInterval(timer);
        setTimeout(() => navigate("/onboard/brand-profile"), 1000);
      }
    }, 1500);

    return () => clearInterval(timer);
  }, [navigate]);

  return (
    <OnboardingLayout step={1} onBack={() => navigate("/onboard/channel")}>
      <div className="flex flex-col items-center justify-center px-6 pt-12 pb-20 max-w-lg mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center mb-10"
        >
          <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-5 relative">
            <Brain className="w-7 h-7 text-primary" />
            <div className="absolute inset-0 rounded-2xl bg-primary/20 animate-ping" />
          </div>
          <h1 className="font-display text-2xl font-extrabold text-foreground">Analyzing your content</h1>
          <p className="text-sm text-muted-foreground mt-2">
            Clypt is scanning your content to build your creator profile.
          </p>
        </motion.div>

        <div className="w-full rounded-xl clypt-glass p-6 mb-6">
          <ProgressTracker
            id="onboard-analysis"
            steps={steps}
            elapsedTime={elapsed}
            className="[&_*]:text-foreground"
          />
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="text-xs text-muted-foreground font-mono"
        >
          This usually takes about 30 seconds
        </motion.p>
      </div>
    </OnboardingLayout>
  );
}
