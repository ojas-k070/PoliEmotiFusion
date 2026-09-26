import type { AnalysisResult, PoliticalCategory } from "@/lib/types";

export interface TextAnalysisRequest {
  text: string;
  category: PoliticalCategory;
}

export interface FileAnalysisRequest {
  file: File;
  category: PoliticalCategory;
  duration?: number;
}

export async function analyzeText(req: TextAnalysisRequest): Promise<AnalysisResult> {
  const response = await fetch("http://127.0.0.1:8000/api/text/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: req.text, category: req.category }),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : "Text analysis failed.";
    throw new Error(message);
  }

  return data as AnalysisResult;
}

export async function analyzeImage(_req: FileAnalysisRequest): Promise<AnalysisResult> {
  throw new Error("This analysis module is not available yet.");
}

export async function analyzeVideo(req: FileAnalysisRequest): Promise<AnalysisResult> {
  const formData = new FormData();

  formData.append("file", req.file);
  formData.append("category", req.category);

  const response = await fetch("http://127.0.0.1:8000/api/video/analyze", {
    method: "POST",
    body: formData,
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : "Video analysis failed.";

    throw new Error(message);
  }

  const emotionMap: Record<string, AnalysisResult["emotion"]> = {
    anger: "Anger",
    joy: "Joy",
    sadness: "Sadness",
    fear: "Fear",
    surprise: "Surprise",
    disgust: "Disgust",
    neutral: "Neutral",
  };

  const probabilities = {
    Anger: Number(data.emotions?.anger ?? 0),
    Joy: Number(data.emotions?.joy ?? 0),
    Sadness: Number(data.emotions?.sadness ?? 0),
    Fear: Number(data.emotions?.fear ?? 0),
    Surprise: Number(data.emotions?.surprise ?? 0),
    Disgust: Number(data.emotions?.disgust ?? 0),
    Neutral: Number(data.emotions?.neutral ?? 0),
  };

  const timeline = Array.isArray(data.temporal_results)
    ? data.temporal_results.map(
        (
          item: {
            start_time?: number;
            end_time?: number;
            dominant_emotion?: string;
            confidence?: number;
          },
          index: number,
        ) => ({
          time:
            typeof item.start_time === "number"
              ? `${item.start_time.toFixed(1)}s`
              : `${index + 1}`,
          emotion: emotionMap[item.dominant_emotion ?? "neutral"] ?? "Neutral",
          confidence: Number(item.confidence ?? 0),
        }),
      )
    : [];

  return {
    id: `video-${Date.now()}`,
    modality: "video",
    emotion: emotionMap[data.dominant_emotion ?? "neutral"] ?? "Neutral",
    confidence: Number(data.confidence ?? 0),
    probabilities,
    timestamp: new Date().toISOString(),
    category: req.category,
    model: "deanngkl/vit-tiny-fer",
    inputLabel: req.file.name,
    summary:
      data.message ??
      "Emotion estimated from visible facial expressions across temporal video windows.",
    timeline,
    frames: Number(data.frames_processed ?? 0),
    duration: Number(data.duration_seconds ?? req.duration ?? 0),
  };
}

export async function analyzeAudio(_req: FileAnalysisRequest): Promise<AnalysisResult> {
  throw new Error("This analysis module is not available yet.");
}