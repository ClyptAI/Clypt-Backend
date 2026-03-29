import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle2, Users, Eye, CalendarDays } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import OnboardingLayout from "@/components/onboarding/OnboardingLayout";
import { onboardingApi } from "@/lib/api";

export default function OnboardChannel() {
  const navigate = useNavigate();
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [found, setFound] = useState(false);
  const [channelData, setChannelData] = useState<any>(null);

  const handleSearch = async () => {
    if (!url.trim()) return;
    setLoading(true);
    try {
      const result = await onboardingApi.resolveChannel(url);
      setChannelData(result);
      setFound(true);
    } catch (err: any) {
      console.error("Channel resolve failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <OnboardingLayout step={0} showBack={false}>
      <div className="flex flex-col items-center justify-center px-6 pt-12 pb-20 max-w-lg mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center mb-10"
        >
          <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-5">
            <svg className="w-7 h-7 text-primary" viewBox="0 0 24 24" fill="currentColor"><path d="M23.5 6.19a3.02 3.02 0 0 0-2.12-2.14C19.5 3.5 12 3.5 12 3.5s-7.5 0-9.38.55A3.02 3.02 0 0 0 .5 6.19 31.7 31.7 0 0 0 0 12a31.7 31.7 0 0 0 .5 5.81 3.02 3.02 0 0 0 2.12 2.14c1.88.55 9.38.55 9.38.55s7.5 0 9.38-.55a3.02 3.02 0 0 0 2.12-2.14A31.7 31.7 0 0 0 24 12a31.7 31.7 0 0 0-.5-5.81zM9.75 15.02V8.98L15.5 12l-5.75 3.02z"/></svg>
          </div>
          <h1 className="font-display text-2xl font-extrabold text-foreground">Connect your channel</h1>
          <p className="text-sm text-muted-foreground mt-2 max-w-sm mx-auto">
            Paste your YouTube channel URL so Clypt can learn your content style.
          </p>
        </motion.div>

        {/* Search input */}
        <div className="w-full rounded-xl clypt-glass p-5 mb-4">
          <div className="flex gap-2">
            <Input
              value={url}
              onChange={(e) => { setUrl(e.target.value); setFound(false); }}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="https://youtube.com/@yourchannel"
              className="flex-1 h-11 bg-background/50 border-border/60 text-foreground placeholder:text-muted-foreground/50 focus-visible:ring-primary/30 rounded-lg font-mono text-sm"
            />
            <Button
              onClick={handleSearch}
              disabled={loading || !url.trim()}
              className="h-11 px-5 rounded-lg font-display font-semibold text-sm"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              ) : (
                "Find"
              )}
            </Button>
          </div>
        </div>

        {/* Channel result card */}
        {found && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full rounded-xl clypt-glass p-5"
          >
            <div className="flex items-start gap-4">
              <img
                src={channelData?.channel?.avatar_url}
                alt={channelData?.channel?.channel_name}
                className="w-14 h-14 rounded-full border-2 border-primary/20 object-cover bg-secondary"
                referrerPolicy="no-referrer"
                crossOrigin="anonymous"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-display font-bold text-foreground text-sm truncate">
                    {channelData?.channel?.channel_name}
                  </h3>
                  <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
                </div>
                <p className="text-xs text-muted-foreground font-mono mb-3">{channelData?.channel?.handle}</p>
                <div className="flex items-center gap-4 flex-wrap">
                  {channelData?.channel?.subscriber_count_label && channelData.channel.subscriber_count_label !== "0" && (
                    <div className="flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-muted-foreground/60" />
                      <span className="text-xs font-mono text-foreground">{channelData.channel.subscriber_count_label}</span>
                    </div>
                  )}
                  {channelData?.channel?.total_views_label && channelData.channel.total_views_label !== "0" && (
                    <div className="flex items-center gap-1.5">
                      <Eye className="w-3.5 h-3.5 text-muted-foreground/60" />
                      <span className="text-xs font-mono text-foreground">{channelData.channel.total_views_label}</span>
                    </div>
                  )}
                  {channelData?.channel?.joined_date_label && (
                    <div className="flex items-center gap-1.5">
                      <CalendarDays className="w-3.5 h-3.5 text-muted-foreground/60" />
                      <span className="text-xs font-mono text-foreground">{channelData.channel.joined_date_label}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <Button
              onClick={() => navigate("/onboard/analyzing", { state: { channelId: channelData.channel.channel_id, channel: channelData.channel } })}
              className="w-full mt-5 h-11 gap-2 font-display font-semibold rounded-lg text-sm"
            >
              Connect this channel
              <ArrowRight className="w-4 h-4" />
            </Button>
          </motion.div>
        )}
      </div>
    </OnboardingLayout>
  );
}
