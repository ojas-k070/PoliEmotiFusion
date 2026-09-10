import type { AnalysisResult, PoliticalCategory } from "@/lib/types";

export interface TextAnalysisRequest {
  text: string;
  category: PoliticalCategory;
}

export interface FileAnalysisRequest {
  file: { name: string; size: number };
  category: PoliticalCategory;
  duration?: number;
}

/**
 * Backend API URL
 */
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Analyze text using the existing Text Emotion API.
 */
export async function analyzeText(
  req: TextAnalysisRequest,
): Promise<AnalysisResult> {
  const response = await fetch(
    `${API_BASE_URL}/api/text/analyze`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: req.text,
        category: req.category,
      }),
    },
  );

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

/**
 * Analyze an image using the EfficientNet-B2 backend API.
 *
 * The backend expects:
 *   file     -> uploaded image
 *   category -> political category
 */
export async function analyzeImage(
  req: FileAnalysisRequest & { file: File },
): Promise<AnalysisResult> {
  const formData = new FormData();

  formData.append("file", req.file);
  formData.append("category", req.category);

  const response = await fetch(
    `${API_BASE_URL}/api/image/analyze`,
    {
      method: "POST",
      body: formData,
    },
  );

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.detail;

    const message =
      typeof detail === "string"
        ? detail
        : detail
          ? JSON.stringify(detail)
          : "Image analysis failed.";

    throw new Error(message);
  }

  return data as AnalysisResult;
}

/**
 * Video analysis placeholder.
 */
export async function analyzeVideo(
  _req: FileAnalysisRequest,
): Promise<AnalysisResult> {
  throw new Error(
    "Video analysis module is not available yet.",
  );
}

/**
 * Audio analysis placeholder.
 */
export async function analyzeAudio(
  _req: FileAnalysisRequest,
): Promise<AnalysisResult> {
  throw new Error(
    "Audio analysis module is not available yet.",
  );
}