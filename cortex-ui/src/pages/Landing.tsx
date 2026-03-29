import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Play, Clock, Hash, Sparkles, Layers, BarChart3, Film } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sampleRuns } from "@/data/mockVideo";

const pipelineSteps = [
  { icon: Sparkles, title: "Ground", desc: "Deterministic GPU extraction — tracks, faces, word timings", num: "01" },
  { icon: Layers, title: "Graph", desc: "Semantic nodes + narrative edges via multimodal reasoning", num: "02" },
  { icon: BarChart3, title: "Score", desc: "Graph-aware ranking — hook, payoff, pacing, arc", num: "03" },
  { icon: Film, title: "Clip", desc: "Render-ready 9:16 configs with framing and captions", num: "04" },
];

export default function Landing() {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [focused, setFocused] = useState(false);
  const navigate = useNavigate();

  const validateAndSubmit = useCallback(() => {
    const ytRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/)|youtu\.be\/)/;
    if (!url.trim()) { setError("Paste a YouTube URL to get started"); return; }
    if (!ytRegex.test(url)) { setError("That doesn't look like a valid YouTube URL"); return; }
    setError("");
    navigate("/run/run-001");
  }, [url, navigate]);

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Layered backgrounds */}
      <div className="absolute inset-0 clypt-grid-bg" />
      <div className="absolute inset-0 clypt-radial-glow" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] rounded-full bg-primary/[0.03] blur-[120px]" />

      {/* Navigation */}
      <header className="relative z-10 h-16 flex items-center justify-between px-6 md:px-10">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="font-display font-extrabold text-primary-foreground text-sm">C</span>
          </div>
          <span className="font-display font-bold text-foreground text-lg tracking-tight">Clypt</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden sm:inline text-xs text-muted-foreground font-mono">v0.1.0</span>
          <div className="h-4 w-px bg-border hidden sm:block" />
          <Button variant="ghost" size="sm" className="text-sm font-display" onClick={() => navigate("/login")}>
            Log in
          </Button>
          <Button size="sm" className="text-sm font-display rounded-lg" onClick={() => navigate("/signup")}>
            Sign up
          </Button>
        </div>
      </header>

      {/* Hero */}
      <div className="relative z-10 flex flex-col items-center justify-center px-6 pt-20 pb-16 md:pt-28 md:pb-20">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="text-center max-w-3xl"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-primary/20 bg-primary/[0.06] text-xs text-primary mb-8"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-glow" />
            Graph-first multimodal intelligence
          </motion.div>

          <h1 className="font-display text-5xl md:text-7xl font-extrabold tracking-tight text-foreground leading-[1.05]">
            Understand video.
            <br />
            <span className="text-primary">Clip smarter.</span>
          </h1>

          <p className="mt-6 text-muted-foreground text-base md:text-lg max-w-lg mx-auto leading-relaxed">
            Clypt builds a semantic graph of your video — who's on screen, how moments connect, 
            and which ones are worth clipping.
          </p>
        </motion.div>

        {/* URL Input */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="mt-12 w-full max-w-xl"
        >
          <div className={`rounded-2xl p-[1px] transition-all duration-300 ${focused ? "bg-gradient-to-r from-primary/50 via-primary/20 to-primary/50" : "bg-border/60"}`}>
            <div className="rounded-[15px] bg-card p-1.5">
              <div className="flex gap-2">
                <Input
                  value={url}
                  onChange={(e) => { setUrl(e.target.value); setError(""); }}
                  onKeyDown={(e) => e.key === "Enter" && validateAndSubmit()}
                  onFocus={() => setFocused(true)}
                  onBlur={() => setFocused(false)}
                  placeholder="Paste a YouTube URL…"
                  className="flex-1 bg-transparent border-0 text-foreground placeholder:text-muted-foreground focus-visible:ring-0 h-12 text-sm font-mono px-4"
                />
                <Button onClick={validateAndSubmit} className="h-12 px-6 gap-2 font-display font-semibold rounded-xl text-sm">
                  Analyze
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
          {error && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-destructive text-xs mt-2.5 ml-3 font-mono"
            >
              {error}
            </motion.p>
          )}
        </motion.div>

        {/* Pipeline strip */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-20 w-full max-w-4xl"
        >
          <div className="flex items-center gap-2 justify-center mb-8">
            <div className="h-px flex-1 max-w-[60px] bg-gradient-to-r from-transparent to-border" />
            <span className="text-[11px] text-muted-foreground uppercase tracking-[0.2em] font-mono">Pipeline</span>
            <div className="h-px flex-1 max-w-[60px] bg-gradient-to-l from-transparent to-border" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {pipelineSteps.map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.5 + i * 0.08, ease: [0.22, 1, 0.36, 1] }}
                className="group relative rounded-xl clypt-glass p-5 hover:border-primary/20 transition-all duration-300"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-data text-[10px] text-muted-foreground/60">{step.num}</span>
                  <step.icon className="w-4 h-4 text-primary/60 group-hover:text-primary transition-colors" />
                </div>
                <h3 className="font-display font-semibold text-sm text-foreground mb-1.5">{step.title}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Sample runs */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.7 }}
        className="relative z-10 px-6 md:px-10 pb-24 max-w-4xl mx-auto"
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="font-display text-base font-bold text-foreground">Sample runs</h2>
          <span className="text-xs text-muted-foreground font-mono">{sampleRuns.length} runs</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {sampleRuns.map((run, i) => (
            <motion.button
              key={run.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.06 }}
              onClick={() => navigate(`/run/${run.id}`)}
              className="group rounded-xl clypt-glass p-5 text-left hover:border-primary/20 transition-all duration-300"
            >
              <div className="flex items-center gap-2 mb-3">
                <span
                  className={`w-2 h-2 rounded-full ${
                    run.status === "completed" ? "bg-clypt-green" : "bg-clypt-amber animate-pulse-glow"
                  }`}
                />
                <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider">{run.status}</span>
              </div>
              <h3 className="font-display text-sm font-semibold text-foreground line-clamp-2 mb-4 group-hover:text-primary transition-colors">
                {run.video.title}
              </h3>
              <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-mono">
                <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{run.video.duration}</span>
                <span className="flex items-center gap-1"><Hash className="w-3 h-3" />{run.metrics.nodes}</span>
                <span className="flex items-center gap-1"><Play className="w-3 h-3" />{run.metrics.clips}</span>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-border/40 py-6 px-6 md:px-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <span className="text-xs text-muted-foreground/60 font-mono">Clypt © 2025</span>
          <span className="text-xs text-muted-foreground/60 font-mono">Built for creators</span>
        </div>
      </footer>
    </div>
  );
}
