import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import OnboardChannel from "./pages/onboarding/OnboardChannel";
import OnboardAnalyzing from "./pages/onboarding/OnboardAnalyzing";
import OnboardBrandProfile from "./pages/onboarding/OnboardBrandProfile";
import OnboardClipPreferences from "./pages/onboarding/OnboardClipPreferences";
import OnboardReady from "./pages/onboarding/OnboardReady";
import RunOverview from "./pages/RunOverview";
import CortexGraph from "./pages/CortexGraph";
import ClipReview from "./pages/ClipReview";
import ArtifactExplorer from "./pages/ArtifactExplorer";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/onboard/channel" element={<OnboardChannel />} />
          <Route path="/onboard/analyzing" element={<OnboardAnalyzing />} />
          <Route path="/onboard/brand-profile" element={<OnboardBrandProfile />} />
          <Route path="/onboard/clip-preferences" element={<OnboardClipPreferences />} />
          <Route path="/onboard/ready" element={<OnboardReady />} />
          <Route path="/run/:id" element={<RunOverview />} />
          <Route path="/run/:id/graph" element={<CortexGraph />} />
          <Route path="/run/:id/clips" element={<ClipReview />} />
          <Route path="/run/:id/artifacts" element={<ArtifactExplorer />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);


export default App;
