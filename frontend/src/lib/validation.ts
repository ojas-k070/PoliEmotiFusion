import type { Modality } from "./types";

export const TEXT_LIMITS = { min: 20, max: 1000 };

export const FILE_RULES: Record<
  Exclude<Modality, "text">,
  { extensions: string[]; accept: string; maxBytes: number; maxDuration?: number }
> = {
  image: {
    extensions: ["jpg", "jpeg", "png", "webp"],
    accept: "image/jpeg,image/png,image/webp",
    maxBytes: 10 * 1024 * 1024,
  },
  video: {
    extensions: ["mp4", "mov", "avi", "webm"],
    accept: "video/mp4,video/quicktime,video/x-msvideo,video/webm",
    maxBytes: 200 * 1024 * 1024,
    maxDuration: 600,
  },
  audio: {
    extensions: ["wav", "mp3", "m4a", "flac"],
    accept: "audio/wav,audio/mpeg,audio/mp4,audio/flac,audio/x-m4a",
    maxBytes: 50 * 1024 * 1024,
    maxDuration: 900,
  },
};

export function validateText(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed) return "Please enter some political text before running an analysis.";
  if (trimmed.length < TEXT_LIMITS.min)
    return `Please enter at least ${TEXT_LIMITS.min} characters for a meaningful analysis.`;
  if (trimmed.length > TEXT_LIMITS.max)
    return `Input exceeds the ${TEXT_LIMITS.max.toLocaleString()} character limit.`;
  return null;
}

export function validateFile(modality: Exclude<Modality, "text">, file: File): string | null {
  const rules = FILE_RULES[modality];
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!rules.extensions.includes(ext))
    return `Unsupported format ".${ext}". Supported formats: ${rules.extensions.join(", ").toUpperCase()}.`;
  if (file.size > rules.maxBytes)
    return `File is too large. Maximum size is ${Math.round(rules.maxBytes / (1024 * 1024))} MB.`;
  return null;
}

export function validateDuration(
  modality: Exclude<Modality, "text">,
  duration: number,
): string | null {
  const max = FILE_RULES[modality].maxDuration;
  if (max && duration > max)
    return `Recording is too long (${Math.round(duration)}s). Maximum supported duration is ${max / 60} minutes.`;
  return null;
}
