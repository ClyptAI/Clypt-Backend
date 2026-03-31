import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, GitBranch, Film, Database, ChevronRight } from "lucide-react";

interface AppShellProps {
  children: React.ReactNode;
  runId?: string;
}

const navItems = [
  { path: "overview", label: "Overview", icon: Activity },
  { path: "graph", label: "Cortex Graph", icon: GitBranch },
  { path: "clips", label: "Clips", icon: Film },
  { path: "artifacts", label: "Artifacts", icon: Database },
];

export function AppShell({ children, runId }: AppShellProps) {
  const location = useLocation();
  const isRunPage = !!runId;
  const currentPath = location.pathname.split("/").pop();

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* Top bar */}
      <header className="h-14 border-b border-border flex items-center px-4 gap-4 shrink-0 clypt-surface">
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-md bg-primary/20 flex items-center justify-center border border-primary/30">
            <GitBranch className="w-4 h-4 text-primary" />
          </div>
          <span className="font-display font-semibold text-foreground tracking-tight">Clypt</span>
        </Link>

        {isRunPage && (
          <>
            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-sm text-muted-foreground font-mono truncate max-w-[300px]">
              Run {runId}
            </span>
            <nav className="ml-6 flex items-center gap-1">
              {navItems.map((item) => {
                const isActive = currentPath === item.path ||
                  (item.path === "overview" && currentPath === runId);
                return (
                  <Link
                    key={item.path}
                    to={`/run/${runId}/${item.path === "overview" ? "" : item.path}`}
                    className={`
                      flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors
                      ${isActive
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                      }
                    `}
                  >
                    <item.icon className="w-3.5 h-3.5" />
                    <span className="hidden md:inline">{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </>
        )}
      </header>

      {/* Content */}
      <main className="flex-1 overflow-hidden">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="h-full"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
