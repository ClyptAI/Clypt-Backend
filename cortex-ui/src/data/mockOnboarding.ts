export const mockChannelResult = {
  channelName: "Theo - t3.gg",
  channelUrl: "https://youtube.com/@t3dotgg",
  subscriberCount: "420K",
  totalViews: "89M",
  avatarUrl: "https://yt3.googleusercontent.com/4NapxEtLjItYASgsA-gVnHkSAGo0PusMsJoVhqfJQQRKJHMmXd5vKzyq9vUO2wO_6plrGEQ9ww=s176-c-k-c0x00ffffff-no-rj",
  bannerUrl: "",
  description: "Full-stack development, opinions, and TypeScript. Previously CEO of Ping.gg.",
  category: "Science & Technology",
  uploadFrequency: "~5 videos/week",
  joinedDate: "2020",
};

export const mockTopShorts = [
  { title: "React Server Components in 60 seconds", views: "2.1M", duration: "58s", likes: "89K" },
  { title: "Stop using useEffect for everything", views: "1.8M", duration: "45s", likes: "76K" },
  { title: "TypeScript trick you NEED to know", views: "1.5M", duration: "52s", likes: "64K" },
  { title: "Why Next.js App Router changes everything", views: "1.3M", duration: "59s", likes: "58K" },
  { title: "The WORST JavaScript pattern", views: "1.1M", duration: "38s", likes: "52K" },
  { title: "Tailwind CSS is NOT what you think", views: "980K", duration: "47s", likes: "45K" },
  { title: "Prisma vs Drizzle in 30 seconds", views: "870K", duration: "30s", likes: "41K" },
  { title: "This React hook is BROKEN", views: "760K", duration: "55s", likes: "38K" },
  { title: "tRPC explained in under a minute", views: "650K", duration: "56s", likes: "34K" },
  { title: "Bun just killed Node.js", views: "540K", duration: "42s", likes: "29K" },
];

export const mockTopVideos = [
  { title: "The T3 Stack in 2024: Complete Guide", views: "1.2M", duration: "32:14", likes: "48K" },
  { title: "The Truth About React Server Components", views: "980K", duration: "28:41", likes: "42K" },
  { title: "Why I Switched from Next.js to Remix", views: "870K", duration: "45:22", likes: "38K" },
  { title: "Full-Stack TypeScript in 2024: The Complete Guide", views: "760K", duration: "1:12:08", likes: "35K" },
  { title: "I Reviewed Vercel's New Pricing Model", views: "650K", duration: "22:15", likes: "31K" },
  { title: "Building a SaaS from Scratch — Live", views: "540K", duration: "2:45:33", likes: "28K" },
  { title: "React 19: Everything You Need to Know", views: "480K", duration: "38:17", likes: "25K" },
  { title: "The Best Database for Your Next Project", views: "420K", duration: "26:44", likes: "22K" },
  { title: "Astro vs Next.js: The Real Comparison", views: "380K", duration: "34:09", likes: "19K" },
  { title: "Why TypeScript is Non-Negotiable in 2024", views: "340K", duration: "18:52", likes: "17K" },
];

export const mockBrandProfile = {
  creatorArchetype: "Educator-Entertainer",
  archetypeDescription: "Blends deep technical expertise with conversational energy and humor. Makes complex topics feel accessible through strong opinions and rapid-fire delivery.",

  dominantMechanisms: {
    humor: { intensity: 0.7, style: "Ironic commentary, self-deprecating asides, meme-aware references" },
    emotion: { intensity: 0.3, style: "Occasional genuine passion about developer experience" },
    social: { intensity: 0.6, style: "Community call-outs, hot takes that drive debate, reaction content" },
    expertise: { intensity: 0.9, style: "Counterintuitive truths, elegant simplification, live coding" },
  },

  audienceSignature: "Developers who want to stay current with the JS/TS ecosystem. They value strong opinions over neutrality, enjoy 'drama' takes on frameworks, and share clips that validate their own tech choices.",

  brandVoice: ["Opinionated", "Fast-paced", "Technically deep", "Meme-literate", "Conversational"],

  recurringThemes: ["TypeScript", "React ecosystem", "Full-stack frameworks", "Developer tooling", "Hot takes on new releases"],

  hookStyle: "Provocative statement or contrarian claim in the first 3 seconds",
  payoffStyle: "Technical revelation or 'I told you so' moment backed by evidence",
};

export const mockClipPreferences = {
  preferredDurationRange: { min: 30, max: 90 },
  targetPlatforms: ["YouTube Shorts", "TikTok", "Twitter/X"],
  tonePreferences: ["Educational", "Entertaining", "Opinionated"],
  avoidTopics: [] as string[],
  captionStyle: "Bold, white with black outline",
  hookImportance: 0.9,
  payoffImportance: 0.8,
};
