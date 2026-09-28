import { createFileRoute } from "@tanstack/react-router";
import {
  AlertCircle,
  CheckCircle2,
  FileAudio,
  Languages,
  Loader2,
  Music,
  Radio,
  Sparkles,
  Volume2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { SpeechIntelligenceResultCard } from "@/components/analysis/SpeechIntelligenceResultCard";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { PoliticalCategory, SpeechIntelligenceResult } from "@/lib/types";
import { validateFile } from "@/lib/validation";
import { analyzeSpeech } from "@/services/analysisService";

export const Route = createFileRoute("/audio")({
  head: () => ({
    meta: [
      { title: "Political Speech Intelligence & Translation — PoliEmotiFusion" },
      {
        name: "description",
        content:
          "Professional multilingual political speech intelligence. Upload speech audio in Hindi, English, Marathi, or Hinglish for automatic language detection, native script transcription, and fluent English translation.",
      },
      { property: "og:title", content: "Political Speech Intelligence & Translation — PoliEmotiFusion" },
      {
        property: "og:description",
        content:
          "Upload political speech audio to generate accurate original transcripts and fluent English translations preserving context, rhetorical style, and intent.",
      },
    ],
  }),
  component: AudioAnalysisPage,
});

type Status = "idle" | "loading" | "success" | "error";

interface ProcessingStep {
  id: string;
  label: string;
  description: string;
}

const PROCESSING_STEPS: ProcessingStep[] = [
  { id: "upload", label: "Uploading Audio...", description: "Transferring recording stream to processing service" },
  { id: "preprocess", label: "Preprocessing Audio...", description: "Converting to 16 kHz mono, normalizing volume, and partitioning chunks" },
  { id: "detect", label: "Detecting Language...", description: "Identifying spoken language (Hindi, English, Marathi, Hinglish, etc.)" },
  { id: "transcribe", label: "Generating Transcript...", description: "Decoding speech into native language text" },
  { id: "translate", label: "Translating to English...", description: "Neural translation preserving context, tone, and rhetorical intent" },
  { id: "completed", label: "Completed.", description: "Finalizing speech intelligence payload" },
];

const AUDIO_SAMPLES = [
  {
    label: "Campaign Rally (Dynamic)",
    filename: "campaign_rally_dynamic.wav",
    category: "Election Campaign" as PoliticalCategory,
  },
  {
    label: "Parliament Address (Measured)",
    filename: "parliament_address_measured.wav",
    category: "Parliament / Legislative Speech" as PoliticalCategory,
  },
  {
    label: "Public Address (Urgent)",
    filename: "political_speech_urgent.wav",
    category: "Public Address" as PoliticalCategory,
  },
  {
    label: "Somber Statement",
    filename: "somber_address.wav",
    category: "Political Speech" as PoliticalCategory,
  },
  {
    label: "Test Speech (Short)",
    filename: "test_speech.wav",
    category: "Political Interview" as PoliticalCategory,
  },
];

function AudioAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [category, setCategory] = useState<PoliticalCategory>("Political Speech");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SpeechIntelligenceResult | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  useEffect(() => {
    if (!file) {
      setAudioUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  // Stepped progress simulation while the server is processing the pipeline
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (status === "loading") {
      setCurrentStepIndex(0);
      interval = setInterval(() => {
        setCurrentStepIndex((prev) => {
          if (prev < PROCESSING_STEPS.length - 2) {
            return prev + 1;
          }
          return prev;
        });
      }, 2500);
    } else if (status === "success") {
      setCurrentStepIndex(PROCESSING_STEPS.length - 1);
    } else {
      setCurrentStepIndex(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [status]);

  const select = (selected: File) => {
    const validation = validateFile("audio", selected);
    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }
    setError(null);
    setFile(selected);
    setStatus("idle");
    setResult(null);
  };

  const clear = () => {
    setFile(null);
    setError(null);
    setResult(null);
    setStatus("idle");
  };

  const loadSample = async (sample: (typeof AUDIO_SAMPLES)[number]) => {
    try {
      const response = await fetch(`/samples/${sample.filename}`);
      if (!response.ok) {
        throw new Error("Could not load sample audio file.");
      }
      const blob = await response.blob();
      const loadedFile = new File([blob], sample.filename, { type: "audio/wav" });
      setFile(loadedFile);
      setCategory(sample.category);
      setError(null);
      setResult(null);
      setStatus("idle");
      toast.success(`Loaded preset sample: ${sample.label}`);
    } catch {
      toast.error("Failed to load preset sample audio.");
    }
  };

  const run = async () => {
    if (!file) {
      const msg = "Please upload or drop an audio recording before running speech intelligence.";
      setError(msg);
      toast.error(msg);
      return;
    }

    const validation = validateFile("audio", file);
    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }

    setError(null);
    setStatus("loading");
    try {
      const res = await analyzeSpeech(file, category);
      setResult(res);
      setStatus("success");
      toast.success(`Speech transcribed in ${res.languageName} & translated to English.`);
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Speech intelligence processing failed.";
      setError(message);
      toast.error(message);
    }
  };

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setError(null);
    setCurrentStepIndex(0);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Multilingual Speech Pipeline · OpenAI Whisper"
        title="Political Speech Intelligence"
        description="Upload a political speech, parliamentary address, public campaign rally, or debate recording (.wav, .mp3, .m4a, .flac). The system automatically detects spoken language (Hindi, English, Marathi, Hinglish), transcribes verbatim in native script, and delivers a high-quality fluent English translation."
      />

      {/* Input Configuration Card */}
      {status !== "success" && (
        <Card className="compactable gap-5 p-6">
          {/* Preset Sample Audio Bar */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <Music className="h-3.5 w-3.5 text-primary" />
              <span>Preset Speech Audio Samples:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {AUDIO_SAMPLES.map((sample) => (
                <Button
                  key={sample.filename}
                  variant="secondary"
                  size="sm"
                  className="text-xs font-medium"
                  onClick={() => loadSample(sample)}
                  disabled={status === "loading"}
                >
                  {sample.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Upload Zone */}
          <UploadZone modality="audio" file={file} onSelect={select} onClear={clear} />
          {error && <p className="text-xs font-medium text-destructive">{error}</p>}

          {/* Audio Player Preview */}
          {audioUrl && (
            <div className="space-y-2 rounded-xl border border-border bg-surface p-4">
              <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
                <Volume2 className="h-4 w-4 text-primary" />
                <span>Audio Playback Preview</span>
              </div>
              <audio controls src={audioUrl} className="h-10 w-full" />
            </div>
          )}

          <CategorySelect value={category} onChange={setCategory} />

          <div className="flex flex-wrap gap-2 pt-2">
            <Button
              onClick={run}
              disabled={!file || status === "loading"}
              className="gap-1.5 shadow-sm"
              size="lg"
            >
              <Languages className="h-4 w-4" />
              {status === "loading" ? "Processing Speech..." : "Analyze Speech Intelligence"}
            </Button>
            <Button
              variant="outline"
              onClick={clear}
              disabled={!file || status === "loading"}
              size="lg"
            >
              Clear
            </Button>
          </div>
        </Card>
      )}

      {/* Live Stepped Progress Indicator */}
      {status === "loading" && (
        <Card className="compactable p-6 space-y-5 border-primary/30 bg-primary/[0.02]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  {PROCESSING_STEPS[currentStepIndex].label}
                </h3>
                <p className="text-xs text-muted-foreground">
                  {PROCESSING_STEPS[currentStepIndex].description}
                </p>
              </div>
            </div>
            <Badge variant="outline" className="text-xs gap-1 border-primary/30 text-primary">
              <Radio className="h-3 w-3 animate-pulse text-primary" />
              Step {currentStepIndex + 1} of {PROCESSING_STEPS.length}
            </Badge>
          </div>

          <Progress
            value={((currentStepIndex + 1) / PROCESSING_STEPS.length) * 100}
            className="h-2"
          />

          <div className="grid gap-2 sm:grid-cols-3 md:grid-cols-6 pt-1">
            {PROCESSING_STEPS.map((step, idx) => {
              const isPast = idx < currentStepIndex;
              const isCurrent = idx === currentStepIndex;

              return (
                <div
                  key={step.id}
                  className={`flex items-center gap-2 rounded-lg border p-2 text-xs transition-colors ${
                    isCurrent
                      ? "border-primary bg-primary/10 text-primary font-semibold"
                      : isPast
                        ? "border-border/80 bg-surface text-muted-foreground"
                        : "border-border/40 opacity-50 text-muted-foreground"
                  }`}
                >
                  {isPast ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                  ) : isCurrent ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
                  ) : (
                    <span className="h-3.5 w-3.5 rounded-full border border-muted-foreground/40 shrink-0" />
                  )}
                  <span className="truncate">{step.label.replace("...", "")}</span>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Error Feedback Card */}
      {status === "error" && (
        <Card className="compactable p-6 space-y-4 border-destructive/40 bg-destructive/5">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-destructive">
                Speech Intelligence Analysis Failed
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {error || "An unexpected error occurred while processing the audio recording."}
              </p>
            </div>
          </div>
          <div className="flex gap-2 pt-1">
            <Button variant="outline" size="sm" onClick={run}>
              Retry Analysis
            </Button>
            <Button variant="ghost" size="sm" onClick={reset}>
              Upload Another Recording
            </Button>
          </div>
        </Card>
      )}

      {/* Success State: Speech Intelligence Results */}
      {status === "success" && result && (
        <SpeechIntelligenceResultCard
          result={result}
          audioUrl={audioUrl}
          onAnalyzeAgain={reset}
        />
      )}
    </div>
  );
}
