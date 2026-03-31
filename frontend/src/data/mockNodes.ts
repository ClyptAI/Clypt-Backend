import type { Node } from "@xyflow/react";

export type SemanticNodeType =
  | "hook"
  | "conflict"
  | "punchline"
  | "payoff"
  | "insight"
  | "topic_shift"
  | "speaker_beat";

export interface NodeComment {
  username: string;
  text: string;
  likes: number;
}

export interface SemanticNodeData {
  label: string;
  type: SemanticNodeType;
  summary: string;
  transcript: string;
  startTime: number;
  endTime: number;
  speaker: string;
  clipWorthy: boolean;
  score: number;
  relatedNodeIds: string[];
  comments: NodeComment[];
  [key: string]: unknown;
}

export const nodeTypeConfig: Record<
  SemanticNodeType,
  { color: string; icon: string; label: string }
> = {
  hook: { color: "var(--clypt-teal)", icon: "🪝", label: "Hook" },
  conflict: { color: "var(--clypt-coral)", icon: "⚡", label: "Conflict" },
  punchline: { color: "var(--clypt-amber)", icon: "💥", label: "Punchline" },
  payoff: { color: "var(--clypt-teal)", icon: "🎯", label: "Payoff" },
  insight: { color: "168 64% 60%", icon: "💡", label: "Insight" },
  topic_shift: { color: "var(--clypt-slate)", icon: "↗️", label: "Topic Shift" },
  speaker_beat: { color: "var(--clypt-mist)", icon: "🎙️", label: "Speaker Beat" },
};

export const mockNodes: Node<SemanticNodeData>[] = [
  {
    id: "n1",
    type: "semantic",
    position: { x: 100, y: 200 },
    data: {
      label: "Cold open provocation",
      type: "hook",
      summary: "Colin opens with a provocative question about whether honesty is actually bad for creators",
      transcript: "\"Here's the thing nobody wants to say out loud — being honest might actually be killing your channel.\"",
      startTime: 0,
      endTime: 12,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.92,
      relatedNodeIds: ["n2", "n5"],
      comments: [
        { username: "@oscarfan", text: "\"the all right all right all right is legendary\"", likes: 14200 },
        { username: "@filmgeek", text: "\"him chasing himself in 10 years is actually profound\"", likes: 8100 },
        { username: "@clipking", text: "\"04:14 is the exact moment he cemented his legacy\"", likes: 5400 },
      ],
    },
  },
  {
    id: "n2",
    type: "semantic",
    position: { x: 350, y: 100 },
    data: {
      label: "Samir's counter-take",
      type: "conflict",
      summary: "Samir pushes back, arguing that honesty is the only sustainable strategy",
      transcript: "\"I actually think it's the opposite. The creators who win long term are the ones who don't play games.\"",
      startTime: 12,
      endTime: 28,
      speaker: "Samir",
      clipWorthy: true,
      score: 0.85,
      relatedNodeIds: ["n1", "n3"],
      comments: [
        { username: "@debatebro", text: "\"samir cooked him here ngl\"", likes: 6300 },
        { username: "@creatorhub", text: "\"this is the take that needed to be said\"", likes: 3800 },
      ],
    },
  },
  {
    id: "n3",
    type: "semantic",
    position: { x: 600, y: 200 },
    data: {
      label: "The algorithm dilemma",
      type: "insight",
      summary: "Discussion about how the algorithm rewards sensationalism over substance",
      transcript: "\"The algorithm doesn't care about truth. It cares about watch time. And that's the tension.\"",
      startTime: 45,
      endTime: 68,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.88,
      relatedNodeIds: ["n2", "n4"],
      comments: [
        { username: "@techrealist", text: "\"the algorithm point is so underrated\"", likes: 9200 },
        { username: "@ytgrinder", text: "\"this is why I stopped chasing trends\"", likes: 4100 },
        { username: "@mediaprof", text: "\"should be taught in every digital media class\"", likes: 2700 },
      ],
    },
  },
  {
    id: "n4",
    type: "semantic",
    position: { x: 850, y: 100 },
    data: {
      label: "MrBeast anecdote",
      type: "speaker_beat",
      summary: "Colin references MrBeast's approach to content and how transparency plays into it",
      transcript: "\"When MrBeast says 'I just make what I would want to watch' — that's honest. But it's also strategic.\"",
      startTime: 72,
      endTime: 95,
      speaker: "Colin",
      clipWorthy: false,
      score: 0.62,
      relatedNodeIds: ["n3", "n5"],
      comments: [
        { username: "@beastfan99", text: "\"MrBeast reference was perfect here\"", likes: 2100 },
      ],
    },
  },
  {
    id: "n5",
    type: "semantic",
    position: { x: 600, y: 380 },
    data: {
      label: "Vulnerability payoff",
      type: "payoff",
      summary: "Samir connects honesty to the specific moments that build real audience trust",
      transcript: "\"The moment you show something real, something that costs you something — that's when people actually care.\"",
      startTime: 120,
      endTime: 145,
      speaker: "Samir",
      clipWorthy: true,
      score: 0.94,
      relatedNodeIds: ["n1", "n6"],
      comments: [
        { username: "@emotionalwreck", text: "\"this hit me in the chest bro\"", likes: 11400 },
        { username: "@realcreator", text: "\"vulnerability IS the content\"", likes: 7600 },
      ],
    },
  },
  {
    id: "n6",
    type: "semantic",
    position: { x: 350, y: 450 },
    data: {
      label: "Burnout admission",
      type: "conflict",
      summary: "Colin admits to recent burnout and questions whether content creation is worth it",
      transcript: "\"I had a week last month where I thought — I'm done. I genuinely didn't want to make another video.\"",
      startTime: 180,
      endTime: 210,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.91,
      relatedNodeIds: ["n5", "n7"],
      comments: [
        { username: "@burnoutgang", text: "\"finally someone admits it\"", likes: 15300 },
        { username: "@mentalhealth", text: "\"creators need to hear this more\"", likes: 8900 },
        { username: "@smallcreator", text: "\"I felt this in my soul\"", likes: 4200 },
      ],
    },
  },
  {
    id: "n7",
    type: "semantic",
    position: { x: 100, y: 400 },
    data: {
      label: "The pivot question",
      type: "topic_shift",
      summary: "Conversation shifts from honesty in content to honesty with yourself as a creator",
      transcript: "\"Okay but let's separate two things — being honest with your audience versus being honest with yourself.\"",
      startTime: 230,
      endTime: 248,
      speaker: "Samir",
      clipWorthy: false,
      score: 0.55,
      relatedNodeIds: ["n6", "n8"],
      comments: [
        { username: "@pivotmaster", text: "\"great transition honestly\"", likes: 1200 },
      ],
    },
  },
  {
    id: "n8",
    type: "semantic",
    position: { x: 250, y: 560 },
    data: {
      label: "Revenue vs. passion",
      type: "conflict",
      summary: "Tension around whether revenue goals corrupt creative honesty",
      transcript: "\"You can't make honest content when you're optimizing for a brand deal. Those two things fight each other.\"",
      startTime: 260,
      endTime: 290,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.86,
      relatedNodeIds: ["n7", "n9"],
      comments: [
        { username: "@bizofcreating", text: "\"the brand deal tension is real\"", likes: 5600 },
        { username: "@indiecreator", text: "\"chose passion over revenue, best decision ever\"", likes: 3400 },
      ],
    },
  },
  {
    id: "n9",
    type: "semantic",
    position: { x: 500, y: 560 },
    data: {
      label: "Creator middle class",
      type: "insight",
      summary: "Insight about the 'creator middle class' and how honesty is more viable when you're not chasing scale",
      transcript: "\"There's a creator middle class emerging. People making $200K a year being completely real. No clickbait needed.\"",
      startTime: 310,
      endTime: 340,
      speaker: "Samir",
      clipWorthy: true,
      score: 0.89,
      relatedNodeIds: ["n8", "n10"],
      comments: [
        { username: "@econnerds", text: "\"creator middle class is the most important concept in this space\"", likes: 7800 },
        { username: "@futureproof", text: "\"$200K being real > $2M being fake\"", likes: 6100 },
      ],
    },
  },
  {
    id: "n10",
    type: "semantic",
    position: { x: 750, y: 450 },
    data: {
      label: "The punchline",
      type: "punchline",
      summary: "Colin delivers a memorable one-liner about the paradox of authenticity",
      transcript: "\"The most dishonest thing you can do is pretend you're being honest. That's the real game.\"",
      startTime: 380,
      endTime: 395,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.95,
      relatedNodeIds: ["n9", "n11"],
      comments: [
        { username: "@quotethis", text: "\"the most dishonest thing is pretending you're honest 🔥\"", likes: 22100 },
        { username: "@philosopher", text: "\"this is literally Baudrillard for YouTubers\"", likes: 4300 },
        { username: "@viralclips", text: "\"clip of the year right here\"", likes: 18700 },
      ],
    },
  },
  {
    id: "n11",
    type: "semantic",
    position: { x: 950, y: 350 },
    data: {
      label: "Audience trust model",
      type: "insight",
      summary: "Framework for how trust compounds over time vs. attention which is borrowed",
      transcript: "\"Attention is a loan. Trust is equity. Every video either adds to or draws from your trust balance.\"",
      startTime: 420,
      endTime: 450,
      speaker: "Samir",
      clipWorthy: true,
      score: 0.93,
      relatedNodeIds: ["n10", "n12"],
      comments: [
        { username: "@trusttheprocess", text: "\"attention is a loan, trust is equity — writing this down\"", likes: 9800 },
        { username: "@startupbro", text: "\"this applies to brands too, not just creators\"", likes: 3200 },
      ],
    },
  },
  {
    id: "n12",
    type: "semantic",
    position: { x: 950, y: 550 },
    data: {
      label: "Personal story — early days",
      type: "speaker_beat",
      summary: "Samir shares a personal story about their first year on YouTube",
      transcript: "\"Our first year, we made 52 videos. I think 3 of them were genuinely honest. The rest were us performing.\"",
      startTime: 480,
      endTime: 510,
      speaker: "Samir",
      clipWorthy: false,
      score: 0.68,
      relatedNodeIds: ["n11", "n13"],
      comments: [
        { username: "@nostalgic", text: "\"52 videos and only 3 honest ones... that's painfully real\"", likes: 5400 },
      ],
    },
  },
  {
    id: "n13",
    type: "semantic",
    position: { x: 750, y: 650 },
    data: {
      label: "Format as honesty",
      type: "insight",
      summary: "Idea that choosing the right format is itself a form of honesty",
      transcript: "\"Podcast format is inherently more honest than scripted. The pauses, the stumbles — you can't fake that.\"",
      startTime: 540,
      endTime: 565,
      speaker: "Colin",
      clipWorthy: false,
      score: 0.72,
      relatedNodeIds: ["n12", "n14"],
      comments: [
        { username: "@podcastlover", text: "\"this is why podcasts > scripted videos\"", likes: 3900 },
        { username: "@formatking", text: "\"the stumbles make it real\"", likes: 2100 },
      ],
    },
  },
  {
    id: "n14",
    type: "semantic",
    position: { x: 500, y: 700 },
    data: {
      label: "Final synthesis",
      type: "payoff",
      summary: "Both agree that the future belongs to creators who are honest AND strategic",
      transcript: "\"You don't have to choose. Be honest and be smart about it. The audience will meet you there.\"",
      startTime: 600,
      endTime: 630,
      speaker: "Samir",
      clipWorthy: true,
      score: 0.9,
      relatedNodeIds: ["n10", "n13"],
      comments: [
        { username: "@wisdomseeker", text: "\"be honest AND smart — that's the whole thesis\"", likes: 6700 },
        { username: "@longform", text: "\"perfect ending to a perfect conversation\"", likes: 4500 },
      ],
    },
  },
  {
    id: "n15",
    type: "semantic",
    position: { x: 250, y: 700 },
    data: {
      label: "Closing hook",
      type: "hook",
      summary: "Colin teases the next episode with a cliffhanger about a creator who went too far with honesty",
      transcript: "\"Next week we're talking to someone who was too honest — and it cost them everything. You don't want to miss it.\"",
      startTime: 650,
      endTime: 670,
      speaker: "Colin",
      clipWorthy: true,
      score: 0.82,
      relatedNodeIds: ["n14"],
      comments: [
        { username: "@hypemaster", text: "\"that cliffhanger tho 😳\"", likes: 7200 },
        { username: "@nextweek", text: "\"already set a reminder for the next episode\"", likes: 3100 },
      ],
    },
  },
];
