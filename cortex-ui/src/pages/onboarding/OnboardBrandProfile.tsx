import { useMemo, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Zap, Heart, Users, GraduationCap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import OnboardingLayout from "@/components/onboarding/OnboardingLayout";
import { mockBrandProfile, mockChannelResult } from "@/data/mockOnboarding";

const mechanismIcons: Record<string, React.ElementType> = {
  humor: Zap,
  emotion: Heart,
  social: Users,
  expertise: GraduationCap,
};

const MECHANISM_LABELS: Record<string, string> = {
  humor: "Humor",
  emotion: "Emotion",
  social: "Social",
  expertise: "Expertise",
};

const HEAT_COLORS = [
  "hsl(0 0% 12%)",
  "hsl(0 90% 45% / 0.25)",
  "hsl(0 90% 45% / 0.50)",
  "hsl(0 90% 45% / 0.75)",
  "hsl(0 90% 45%)",
];

function generateMechanismCells(intensity: number, count: number): number[] {
  const cells: number[] = [];
  for (let i = 0; i < count; i++) {
    const base = Math.round(intensity * 4);
    const noise = Math.random() > 0.7 ? (Math.random() > 0.5 ? 1 : -1) : 0;
    cells.push(Math.max(0, Math.min(4, base + noise)));
  }
  return cells;
}

export default function OnboardBrandProfile() {
  const navigate = useNavigate();
  const location = useLocation();
  const { creatorId, profile, channel } = (location.state as any) || {};

  // Map API profile to UI shape, falling back to mock data
  const brandProfile = profile
    ? {
        creatorArchetype: profile.creator_archetype ?? mockBrandProfile.creatorArchetype,
        archetypeDescription: profile.archetype_description ?? mockBrandProfile.archetypeDescription,
        dominantMechanisms: profile.dominant_mechanisms ?? mockBrandProfile.dominantMechanisms,
        brandVoice: profile.brand_voice ?? mockBrandProfile.brandVoice,
        hookStyle: profile.hook_style ?? mockBrandProfile.hookStyle,
        payoffStyle: profile.payoff_style ?? mockBrandProfile.payoffStyle,
        audienceSignature: profile.audience_signature ?? mockBrandProfile.audienceSignature,
      }
    : mockBrandProfile;

  // Map snake_case API channel data to UI shape, falling back to mock
  const channelData = channel
    ? {
        channelName: channel.channel_name ?? channel.channelName ?? "",
        avatarUrl: channel.avatar_url ?? channel.avatarUrl ?? "",
        subscriberCount: channel.subscriber_count_label ?? channel.subscriberCount ?? "",
        totalViews: channel.total_views_label ?? channel.totalViews ?? "",
        uploadFrequency: channel.upload_frequency_label ?? channel.uploadFrequency ?? "",
        joinedDate: channel.joined_date_label ?? channel.joinedDate ?? "",
        category: channel.category ?? "",
        description: channel.description ?? "",
      }
    : mockChannelResult;

  const { dominantMechanisms } = brandProfile;

  const mechanismKeys = Object.keys(dominantMechanisms) as Array<
    keyof typeof dominantMechanisms
  >;

  // Generate heatmap data for each mechanism
  const heatData = useMemo(
    () =>
      mechanismKeys.map((key) => ({
        key,
        label: MECHANISM_LABELS[key] || key,
        cells: generateMechanismCells(dominantMechanisms[key].intensity, 12),
      })),
    [],
  );

  return (
    <OnboardingLayout step={2} onBack={() => navigate("/onboard/analyzing")}>
      <div className="flex flex-col items-center px-6 pt-8 pb-20 max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center mb-8"
        >
          <h1 className="font-display text-2xl font-extrabold text-foreground">
            Your brand profile
          </h1>
          <p className="text-sm text-muted-foreground mt-2">
            Here's what Clypt learned about your content style.
          </p>
        </motion.div>

        {/* Channel stats — full width */}
        <div className="w-full rounded-xl clypt-glass p-5 mb-4">
          <div className="flex items-center gap-4 mb-4">
            <img
              src={channelData.avatarUrl}
              alt={channelData.channelName}
              className="w-10 h-10 rounded-full border border-primary/20 object-cover bg-secondary"
              referrerPolicy="no-referrer"
              crossOrigin="anonymous"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
            <div>
              <h2 className="font-display text-sm font-bold text-foreground">{channelData.channelName}</h2>
              <p className="text-xs text-muted-foreground font-mono">{channelData.category} · Joined {channelData.joinedDate}</p>
            </div>
          </div>
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: "Subscribers", value: channelData.subscriberCount },
              { label: "Total Views", value: channelData.totalViews },
              { label: "Uploads", value: channelData.uploadFrequency },
              { label: "Category", value: channelData.category },
            ].map((stat) => (
              <div key={stat.label} className="rounded-lg bg-background/40 border border-border/40 p-3 text-center">
                <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-1">{stat.label}</p>
                <p className="text-base font-display font-bold text-foreground">{stat.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Two-column layout */}
        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* LEFT COLUMN */}
          <div className="flex flex-col gap-4">
            {/* Archetype card */}
            <div className="rounded-xl clypt-glass p-6 flex-1">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-2">
                Creator archetype
              </p>
              <h2 className="font-display text-lg font-bold text-primary mb-2">
                {brandProfile.creatorArchetype}
              </h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {brandProfile.archetypeDescription}
              </p>
            </div>

            {/* Brand voice tags */}
            <div className="rounded-xl clypt-glass p-6">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-3">
                Brand voice
              </p>
              <div className="flex flex-wrap gap-2">
                {brandProfile.brandVoice.map((tag) => (
                  <Badge
                    key={tag}
                    variant="secondary"
                    className="bg-primary/10 text-primary border-primary/20 font-mono text-xs"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Hook / Payoff */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl clypt-glass p-5">
                <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-2">
                  Hook style
                </p>
                <p className="text-xs text-foreground leading-relaxed">
                  {brandProfile.hookStyle}
                </p>
              </div>
              <div className="rounded-xl clypt-glass p-5">
                <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-2">
                  Payoff style
                </p>
                <p className="text-xs text-foreground leading-relaxed">
                  {brandProfile.payoffStyle}
                </p>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div className="flex flex-col gap-4">
            {/* Heatmap */}
            <div className="rounded-xl clypt-glass p-5">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-1">
                Content Mechanism Intensity
              </p>
              <p className="text-xs text-muted-foreground mb-4">
                Distribution across recent videos
              </p>
              <div className="space-y-3">
                {heatData.map((mechanism) => (
                  <div key={mechanism.key} className="flex items-center gap-3">
                    <span className="text-xs font-mono text-muted-foreground w-16 shrink-0 capitalize">
                      {mechanism.label}
                    </span>
                    <div className="flex gap-[3px] flex-1">
                      {mechanism.cells.map((level, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, scale: 0.5 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: i * 0.03 }}
                          className="w-5 h-5 rounded-[3px] cursor-default"
                          style={{ backgroundColor: HEAT_COLORS[level] }}
                          title={`${mechanism.label}: Level ${level}/4`}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-4">
                <span className="text-[10px] text-muted-foreground/60 font-mono">Low</span>
                {HEAT_COLORS.map((color, i) => (
                  <div key={i} className="w-3.5 h-3.5 rounded-[2px]" style={{ backgroundColor: color }} />
                ))}
                <span className="text-[10px] text-muted-foreground/60 font-mono">High</span>
              </div>
            </div>

            {/* Dominant mechanisms */}
            <div className="rounded-xl clypt-glass p-6 flex-1">
              <p className="text-[10px] text-muted-foreground/60 font-mono uppercase tracking-wider mb-4">
                Dominant mechanisms
              </p>
              <div className="space-y-4">
                {Object.entries(dominantMechanisms).map(([key, val]) => {
                  const Icon = mechanismIcons[key] || Zap;
                  return (
                    <div key={key} className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <Icon className="w-4 h-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-sm font-display font-semibold text-foreground capitalize">{key}</span>
                          <span className="text-xs font-mono text-primary">{Math.round(val.intensity * 100)}%</span>
                        </div>
                        <div className="h-1 rounded-full bg-border overflow-hidden mb-1.5">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${val.intensity * 100}%` }}
                            transition={{ duration: 0.8, delay: 0.2 }}
                            className="h-full rounded-full bg-primary"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground">{val.style}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        <Button
          onClick={() => navigate("/onboard/clip-preferences", { state: { creatorId, profile, channel } })}
          className="w-full max-w-sm h-11 gap-2 font-display font-semibold rounded-lg text-sm"
        >
          Looks good, continue
          <ArrowRight className="w-4 h-4" />
        </Button>
      </div>
    </OnboardingLayout>
  );
}
