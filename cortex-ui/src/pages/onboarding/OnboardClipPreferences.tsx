import { useState, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { ParameterSlider } from "@/components/tool-ui/parameter-slider";
import { OptionList } from "@/components/tool-ui/option-list";
import type { SliderValue } from "@/components/tool-ui/parameter-slider";
import type { OptionListSelection } from "@/components/tool-ui/option-list";
import OnboardingLayout from "@/components/onboarding/OnboardingLayout";
import { mockClipPreferences } from "@/data/mockOnboarding";
import { creatorApi } from "@/lib/api";

export default function OnboardClipPreferences() {
  const navigate = useNavigate();
  const location = useLocation();
  const { creatorId } = (location.state as any) || {};
  const [durationMin, setDurationMin] = useState(mockClipPreferences.preferredDurationRange.min);
  const [durationMax, setDurationMax] = useState(mockClipPreferences.preferredDurationRange.max);
  const [hookImportance, setHookImportance] = useState(mockClipPreferences.hookImportance * 100);
  const [payoffImportance, setPayoffImportance] = useState(mockClipPreferences.payoffImportance * 100);
  const [platforms, setPlatforms] = useState<OptionListSelection>(
    mockClipPreferences.targetPlatforms.map((p) => p.toLowerCase().replace(/[\s\/]/g, "-"))
  );
  const [tones, setTones] = useState<OptionListSelection>(
    mockClipPreferences.tonePreferences.map((t) => t.toLowerCase())
  );
  const [autoCaptions, setAutoCaptions] = useState(true);

  const handleDurationChange = useCallback((values: SliderValue[]) => {
    const min = values.find((v) => v.id === "min-duration");
    const max = values.find((v) => v.id === "max-duration");
    if (min) setDurationMin(min.value);
    if (max) setDurationMax(max.value);
  }, []);

  const handleImportanceChange = useCallback((values: SliderValue[]) => {
    const hook = values.find((v) => v.id === "hook");
    const payoff = values.find((v) => v.id === "payoff");
    if (hook) setHookImportance(hook.value);
    if (payoff) setPayoffImportance(payoff.value);
  }, []);

  return (
    <OnboardingLayout step={3} onBack={() => navigate("/onboard/brand-profile")}>
      <div className="flex flex-col items-center px-6 pt-8 pb-20 max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-5">
            <Settings2 className="w-7 h-7 text-primary" />
          </div>
          <h1 className="font-display text-2xl font-extrabold text-foreground">Clip preferences</h1>
          <p className="text-sm text-muted-foreground mt-2">Fine-tune how Clypt selects and scores your clips.</p>
        </motion.div>

        {/* Two-column layout */}
        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* LEFT COLUMN — Sliders */}
          <div className="flex flex-col gap-4">
            <div className="rounded-xl clypt-glass p-6 [&_*]:text-foreground">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-4">Clip duration</p>
              <ParameterSlider
                id="clip-duration"
                sliders={[
                  { id: "min-duration", label: "Min Duration", min: 15, max: 180, step: 5, value: durationMin, unit: "s" },
                  { id: "max-duration", label: "Max Duration", min: 15, max: 180, step: 5, value: durationMax, unit: "s" },
                ]}
                onChange={handleDurationChange}
              />
            </div>

            <div className="rounded-xl clypt-glass p-6 [&_*]:text-foreground">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-4">Scoring weights</p>
              <ParameterSlider
                id="clip-importance"
                sliders={[
                  { id: "hook", label: "Hook Importance", min: 0, max: 100, step: 5, value: hookImportance, unit: "%" },
                  { id: "payoff", label: "Payoff Importance", min: 0, max: 100, step: 5, value: payoffImportance, unit: "%" },
                ]}
                onChange={handleImportanceChange}
              />
            </div>

            {/* Auto captions */}
            <div className="rounded-xl clypt-glass p-5 flex items-center justify-between">
              <div>
                <p className="text-sm font-display font-semibold text-foreground">Auto-captions</p>
                <p className="text-xs text-muted-foreground">Bold white with black outline</p>
              </div>
              <Switch checked={autoCaptions} onCheckedChange={setAutoCaptions} />
            </div>
          </div>

          {/* RIGHT COLUMN — Selections */}
          <div className="flex flex-col gap-4">
            <div className="rounded-xl clypt-glass p-6 [&_*]:text-foreground">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-3">Target platforms</p>
              <OptionList
                id="target-platforms"
                selectionMode="multi"
                defaultValue={platforms}
                onChange={(val) => setPlatforms(val)}
                options={[
                  { id: "youtube-shorts", label: "YouTube Shorts", description: "Vertical, up to 60s" },
                  { id: "tiktok", label: "TikTok", description: "Vertical, up to 3min" },
                  { id: "twitter-x", label: "Twitter/X", description: "Horizontal or vertical, up to 2:20" },
                  { id: "instagram-reels", label: "Instagram Reels", description: "Vertical, up to 90s" },
                  { id: "linkedin", label: "LinkedIn", description: "Horizontal, up to 10min" },
                ]}
              />
            </div>

            <div className="rounded-xl clypt-glass p-6 flex-1 [&_*]:text-foreground">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-3">Preferred tone</p>
              <OptionList
                id="tone-preferences"
                selectionMode="multi"
                defaultValue={tones}
                onChange={(val) => setTones(val)}
                options={[
                  { id: "educational", label: "Educational" },
                  { id: "entertaining", label: "Entertaining" },
                  { id: "opinionated", label: "Opinionated" },
                  { id: "inspirational", label: "Inspirational" },
                  { id: "dramatic", label: "Dramatic" },
                  { id: "casual", label: "Casual" },
                  { id: "professional", label: "Professional" },
                ]}
              />
            </div>
          </div>
        </div>

        <Button
          onClick={async () => {
            if (creatorId) {
              try {
                await creatorApi.savePreferences(creatorId, {
                  preferred_duration_range: { min_seconds: durationMin, max_seconds: durationMax },
                  target_platforms: Array.isArray(platforms) ? platforms : [],
                  tone_preferences: Array.isArray(tones) ? tones : [],
                  caption_style: autoCaptions ? "Bold, white with black outline" : "none",
                  hook_importance: hookImportance / 100,
                  payoff_importance: payoffImportance / 100,
                });
              } catch (err) {
                console.error("Failed to save preferences:", err);
              }
            }
            navigate("/onboard/ready", { state: { creatorId } });
          }}
          className="w-full max-w-sm h-11 gap-2 font-display font-semibold rounded-lg text-sm"
        >
          Save preferences
          <ArrowRight className="w-4 h-4" />
        </Button>
      </div>
    </OnboardingLayout>
  );
}
