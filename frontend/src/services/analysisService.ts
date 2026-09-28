import type { AnalysisResult, PoliticalCategory, SpeechIntelligenceResult } from "@/lib/types";

export interface TextAnalysisRequest {
  text: string;
  category: PoliticalCategory;
}

export interface FileAnalysisRequest {
  file: { name: string; size: number };
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

export async function analyzeVideo(_req: FileAnalysisRequest): Promise<AnalysisResult> {
  throw new Error("This analysis module is not available yet.");
}

export interface SpeechAnalysisRequest {
  file: File;
  category: PoliticalCategory;
}

export async function analyzeSpeech(
  fileOrReq: File | SpeechAnalysisRequest,
  maybeCategory?: PoliticalCategory,
): Promise<SpeechIntelligenceResult> {
  let file: File;
  let category: PoliticalCategory = "Other";

  if (fileOrReq instanceof File) {
    file = fileOrReq;
    category = maybeCategory ?? "Other";
  } else {
    file = fileOrReq.file;
    category = fileOrReq.category ?? "Other";
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("category", category);

  const response = await fetch("http://127.0.0.1:8000/api/audio/analyze", {
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
          : "Speech intelligence processing failed.";
    throw new Error(message);
  }

  return data as SpeechIntelligenceResult;
}

/** Backward-compatible alias for existing callers */
export const analyzeAudio = analyzeSpeech;
