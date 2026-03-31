export type FramingType = "single_person" | "shared_two_shot" | "split_layout";

export interface ClipCandidate {
  id: string;
  rank: number;
  title: string;
  startTime: number;
  endTime: number;
  duration: number;
  score: number;
  transcript: string;
  justification: string;
  framingType: FramingType;
  speaker: string;
  scores: {
    hook: number;
    payoff: number;
    pacing: number;
    narrativeArc: number;
    clipWorthiness: number;
  };
  nodeIds: string[];
  pinned: boolean;
  bestCut: boolean;
}

export const framingLabels: Record<FramingType, string> = {
  single_person: "Single-Person Framing",
  shared_two_shot: "Shared Two-Shot",
  split_layout: "Explicit Split Layout",
};

export const mockClips: ClipCandidate[] = [
  {
    id: "clip-1",
    rank: 1,
    title: "The Authenticity Paradox",
    startTime: 370,
    endTime: 400,
    duration: 30,
    score: 0.95,
    transcript: "\"The most dishonest thing you can do is pretend you're being honest. That's the real game.\" — Colin delivers the episode's sharpest take, immediately validated by Samir's reaction.",
    justification: "Peak narrative density: combines the episode's central conflict with a memorable one-liner. Strong hook, immediate payoff, clean single-speaker delivery.",
    framingType: "single_person",
    speaker: "Colin",
    scores: { hook: 0.93, payoff: 0.97, pacing: 0.91, narrativeArc: 0.94, clipWorthiness: 0.95 },
    nodeIds: ["n10"],
    pinned: false,
    bestCut: true,
  },
  {
    id: "clip-2",
    rank: 2,
    title: "Attention vs. Trust",
    startTime: 415,
    endTime: 455,
    duration: 40,
    score: 0.93,
    transcript: "\"Attention is a loan. Trust is equity. Every video either adds to or draws from your trust balance.\" — Samir introduces a framework that reframes the entire creator economy.",
    justification: "Novel framework delivery with strong metaphor. Highly quotable, clean pacing, and immediately applicable to any creator's strategy.",
    framingType: "single_person",
    speaker: "Samir",
    scores: { hook: 0.88, payoff: 0.95, pacing: 0.94, narrativeArc: 0.92, clipWorthiness: 0.93 },
    nodeIds: ["n11"],
    pinned: false,
    bestCut: false,
  },
  {
    id: "clip-3",
    rank: 3,
    title: "The Burnout Moment",
    startTime: 175,
    endTime: 215,
    duration: 40,
    score: 0.91,
    transcript: "\"I had a week last month where I thought — I'm done. I genuinely didn't want to make another video.\" — Colin's raw admission shifts the entire conversation.",
    justification: "Highest emotional authenticity in the episode. Vulnerability creates audience connection. Best as single-person framing to maintain intimacy.",
    framingType: "single_person",
    speaker: "Colin",
    scores: { hook: 0.90, payoff: 0.88, pacing: 0.85, narrativeArc: 0.95, clipWorthiness: 0.91 },
    nodeIds: ["n6"],
    pinned: false,
    bestCut: false,
  },
  {
    id: "clip-4",
    rank: 4,
    title: "The Creator Middle Class",
    startTime: 305,
    endTime: 345,
    duration: 40,
    score: 0.89,
    transcript: "\"There's a creator middle class emerging. People making $200K a year being completely real. No clickbait needed.\" — Samir offers an optimistic counter to the revenue/passion conflict.",
    justification: "Data-driven insight with contrarian optimism. Strong standalone clip with clear takeaway. Two-shot captures both reactions.",
    framingType: "shared_two_shot",
    speaker: "Samir",
    scores: { hook: 0.82, payoff: 0.91, pacing: 0.90, narrativeArc: 0.87, clipWorthiness: 0.89 },
    nodeIds: ["n9"],
    pinned: false,
    bestCut: false,
  },
  {
    id: "clip-5",
    rank: 5,
    title: "The Opening Provocation",
    startTime: 0,
    endTime: 30,
    duration: 30,
    score: 0.87,
    transcript: "\"Here's the thing nobody wants to say out loud — being honest might actually be killing your channel.\" \"I actually think it's the opposite.\" — The opening exchange sets up the entire episode's tension.",
    justification: "Classic hook-and-counter pattern. Split layout captures both speakers' energy. Strong cold-open for short-form.",
    framingType: "split_layout",
    speaker: "Colin & Samir",
    scores: { hook: 0.96, payoff: 0.72, pacing: 0.88, narrativeArc: 0.82, clipWorthiness: 0.87 },
    nodeIds: ["n1", "n2"],
    pinned: false,
    bestCut: false,
  },
  {
    id: "clip-6",
    rank: 6,
    title: "Honest AND Strategic",
    startTime: 595,
    endTime: 635,
    duration: 40,
    score: 0.85,
    transcript: "\"You don't have to choose. Be honest and be smart about it. The audience will meet you there.\" — The final synthesis resolves the episode's central tension.",
    justification: "Clean narrative resolution. Works as a standalone motivational clip. Two-shot framing captures mutual agreement energy.",
    framingType: "shared_two_shot",
    speaker: "Samir",
    scores: { hook: 0.70, payoff: 0.94, pacing: 0.86, narrativeArc: 0.90, clipWorthiness: 0.85 },
    nodeIds: ["n14"],
    pinned: false,
    bestCut: false,
  },
];
