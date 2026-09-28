export type Modality = "text" | "image" | "video" | "audio";

export const EMOTIONS = [
  "Anger",
  "Joy",
  "Sadness",
  "Fear",
  "Surprise",
  "Disgust",
  "Neutral",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export const POLITICAL_CATEGORIES = [
  "Political Speech",
  "Election Campaign",
  "Political Debate",
  "Parliament / Legislative Speech",
  "Political Interview",
  "Political Social Media",
  "Political News",
  "Public Address",
  "Other",
] as const;

export type PoliticalCategory = (typeof POLITICAL_CATEGORIES)[number];

export type TimelinePoint = { time: string; emotion: Emotion; confidence: number };

export interface AnalysisResult {
  id: string;
  modality: Modality;
  emotion: Emotion;
  confidence: number;
  probabilities: Record<Emotion, number>;
  timestamp: string;
  category: PoliticalCategory;
  model: string;
  inputLabel: string;
  summary: string;
  /** video only */
  timeline?: TimelinePoint[] | undefined;
  frames?: number | undefined;
  /** image only */
  facesDetected?: number | undefined;
  /** audio + video */
  duration?: number | undefined;
  /** audio only */
  waveform?: number[] | undefined;
}

export interface NonPoliticalImageResult {
  status: "not_political_content";
  political_score: number;
  threshold: number;
  message: string;
}

export interface HistoryRecord {
  id: string;
  date: string;
  modality: Modality;
  inputLabel: string;
  emotion: Emotion;
  confidence: number;
  category: PoliticalCategory;
  status: "Completed" | "Failed";
  result?: AnalysisResult | undefined;
}
