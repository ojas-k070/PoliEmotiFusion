import type { Modality, PoliticalCategory } from "@/lib/types";

export const MODEL_LABELS: Record<Modality, string> = {
  text: "Text Pipeline · DistilRoBERTa Emotion",
  image: "Vision Pipeline · Image Emotion",
  video: "Video Pipeline · Video Emotion",
  audio: "Speech Pipeline · Audio Emotion",
};

export const LOADING_MESSAGES: Record<Modality, string[]> = {
  text: ["Validating input...", "Processing text...", "Scoring emotion classes..."],
  image: ["Reading image...", "Detecting facial features...", "Scoring emotion classes..."],
  video: ["Sampling frames...", "Processing video frames...", "Building emotion timeline..."],
  audio: ["Decoding audio...", "Analyzing speech characteristics...", "Scoring emotion classes..."],
};

export const TEXT_EXAMPLES: { label: string; category: PoliticalCategory; text: string }[] = [
  {
    label: "Political Speech",
    category: "Political Speech",
    text: "For too long our citizens have been asked to accept less while being promised more. Today we say clearly: that arrangement ends. We will rebuild the institutions that serve the public, and we will do it in the open, with every rupee accounted for.",
  },
  {
    label: "Debate Statement",
    category: "Political Debate",
    text: "My colleague has had four years to present a plan and has presented none. The numbers he quoted tonight do not appear in any published budget document, and the public deserves an explanation rather than another slogan.",
  },
  {
    label: "Social Media Post",
    category: "Political Social Media",
    text: "Third power cut this week in our ward and still no response from the municipal office. People here are tired of filing complaints that go nowhere. We are documenting everything.",
  },
  {
    label: "News Statement",
    category: "Political News",
    text: "The committee confirmed today that the legislative session will be extended by two weeks to complete review of the pending amendments. Members from both benches described the discussion as constructive.",
  },
];
