const API_BASE = "/api/v1";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...options.headers as Record<string, string> },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// Auth
export const authApi = {
  signup: (email: string, password: string, display_name: string) =>
    request<{ token: string; user: any }>("/auth/signup", {
      method: "POST",
      body: JSON.stringify({ email, password, display_name }),
    }),
  login: (email: string, password: string) =>
    request<{ token: string; user: any }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<{ user: any }>("/auth/me"),
  logout: () => request<{ logged_out: boolean }>("/auth/logout", { method: "POST" }),
};

// Onboarding
export const onboardingApi = {
  resolveChannel: (query: string) =>
    request<{ channel: any; recent_shorts: any[]; recent_videos: any[] }>("/onboarding/channel/resolve", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),
  analyzeChannel: (channel_id: string) =>
    request<{ job_id: string; status: string }>("/onboarding/channel/analyze", {
      method: "POST",
      body: JSON.stringify({ channel_id }),
    }),
  getAnalysisJob: (jobId: string) =>
    request<any>(`/onboarding/channel/analyze/${jobId}`),
};

// Creator
export const creatorApi = {
  getProfile: (creatorId: string) =>
    request<any>(`/creators/${creatorId}/profile`),
  getPreferences: (creatorId: string) =>
    request<any>(`/creators/${creatorId}/preferences`),
  savePreferences: (creatorId: string, prefs: any) =>
    request<any>(`/creators/${creatorId}/preferences`, {
      method: "PUT",
      body: JSON.stringify(prefs),
    }),
};

// Retrieve
export const retrieveApi = {
  retrieveClip: (runId: string, payload: any) =>
    request<any>(`/runs/${runId}/clips/retrieve`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
