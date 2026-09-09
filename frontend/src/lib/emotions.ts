import { EMOTIONS, type Emotion } from "./types";

export const EMOTION_COLORS: Record<Emotion, string> = {
  Anger: "var(--emotion-anger)",
  Joy: "var(--emotion-joy)",
  Sadness: "var(--emotion-sadness)",
  Fear: "var(--emotion-fear)",
  Surprise: "var(--emotion-surprise)",
  Disgust: "var(--emotion-disgust)",
  Neutral: "var(--emotion-neutral)",
};

export const EMOTION_TEXT_CLASS: Record<Emotion, string> = {
  Anger: "text-emotion-anger",
  Joy: "text-emotion-joy",
  Sadness: "text-emotion-sadness",
  Fear: "text-emotion-fear",
  Surprise: "text-emotion-surprise",
  Disgust: "text-emotion-disgust",
  Neutral: "text-emotion-neutral",
};

export function emptyDistribution(): Record<Emotion, number> {
  return EMOTIONS.reduce(
    (acc, e) => ({ ...acc, [e]: 0 }),
    {} as Record<Emotion, number>,
  );
}

/** Normalises weights into percentages summing to 100. */
export function buildDistribution(weights: Partial<Record<Emotion, number>>): Record<Emotion, number> {
  const filled = EMOTIONS.map((e) => Math.max(0.01, weights[e] ?? 0.01));
  const total = filled.reduce((a, b) => a + b, 0);
  const raw = filled.map((v) => (v / total) * 100);
  const rounded = raw.map((v) => Math.round(v * 10) / 10);
  const drift = Math.round((100 - rounded.reduce((a, b) => a + b, 0)) * 10) / 10;
  const maxIdx = rounded.indexOf(Math.max(...rounded));
  rounded[maxIdx] = Math.round(((rounded[maxIdx] ?? 0) + drift) * 10) / 10;
  return EMOTIONS.reduce(
    (acc, e, i) => ({ ...acc, [e]: rounded[i] ?? 0 }),
    {} as Record<Emotion, number>,
  );
}

export function topEmotion(probabilities: Record<Emotion, number>): Emotion {
  const sorted = (Object.entries(probabilities) as [Emotion, number][]).sort(
    (a, b) => b[1] - a[1],
  );
  return sorted[0]?.[0] ?? "Neutral";
}

export function formatConfidence(value: number) {
  return `${value.toFixed(1)}%`;
}

export function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
