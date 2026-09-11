import { createFileRoute } from "@tanstack/react-router";
import { Mic, Volume2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { AudioWaveform } from "@/components/analysis/AudioWaveform";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { ResultCard } from "@/components/analysis/ResultCard";
import { ErrorState, LoadingState } from "@/components/analysis/StateCards";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { addAnalysis } from "@/lib/store";
import type { AnalysisResult, PoliticalCategory } from "@/lib/types";
import { validateFile } from "@/lib/validation";
import { analyzeAudio } from "@/services/analysisService";

export const Route = createFileRoute("/audio")({
  head: () => ({
    meta: [
      { title: "Political Audio Emotion Analysis — PoliEmotiFusion" },
      {
        name: "description",
        content:
          "Analyze speech emotions from political speeches, addresses, debates, and interview audio recordings.",
      },
      { property: "og:title", content: "Political Audio Emotion Analysis — PoliEmotiFusion" },
      {
        property: "og:description",
        content:
          "Upload political audio recordings to detect speech emotion signals with probability distribution and waveforms.",
      },
    ],
  }),
  component: AudioAnalysisPage,
});

type Status = "idle" | "loading" | "success" | "error";

function AudioAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [category, setCategory] = useState<PoliticalCategory>("Political Speech");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

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

  const run = async () => {
    if (!file) {
      const msg = "Please upload or drop an audio recording before analyzing.";
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
      const res = await analyzeAudio(file, category);
      setResult(res);
      addAnalysis(res);
      setStatus("success");
      toast.success(`Detected emotion: ${res.emotion}`);
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Audio analysis failed.";
      toast.error(message);
    }
  };

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Speech pipeline · Audio Emotion"
        title="Political Audio Emotion Analysis"
        description="Upload a political speech, campaign address, debate excerpt, or interview recording to analyze vocal emotional signals and acoustic tone."
      />

      <Card className="compactable gap-5 p-6">
        <UploadZone modality="audio" file={file} onSelect={select} onClear={clear} />
        {error && <p className="text-xs font-medium text-destructive">{error}</p>}

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

        <div className="flex flex-wrap gap-2">
          <Button
            onClick={run}
            disabled={!file || status === "loading"}
            className="gap-1.5"
          >
            <Mic className="h-4 w-4" />
            {status === "loading" ? "Analyzing Audio..." : "Analyze Audio Emotion"}
          </Button>
          <Button
            variant="outline"
            onClick={clear}
            disabled={!file || status === "loading"}
          >
            Clear
          </Button>
        </div>
      </Card>

      {status === "loading" && <LoadingState modality="audio" />}
      {status === "error" && (
        <ErrorState onRetry={run} onReset={reset} resetLabel="Upload another file" />
      )}
      {status === "success" && result && (
        <ResultCard
          result={result}
          onAnalyzeAgain={reset}
          preview={
            result.waveform && result.waveform.length > 0 ? (
              <div className="space-y-2 rounded-xl border border-border bg-surface p-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold">Audio Waveform</p>
                  {result.duration && (
                    <span className="text-xs text-muted-foreground">
                      {result.duration.toFixed(1)}s
                    </span>
                  )}
                </div>
                <AudioWaveform data={result.waveform} />
                {audioUrl && (
                  <div className="pt-2">
                    <audio controls src={audioUrl} className="h-9 w-full" />
                  </div>
                )}
              </div>
            ) : audioUrl ? (
              <div className="space-y-2 rounded-xl border border-border bg-surface p-4">
                <p className="text-sm font-semibold">Recording Playback</p>
                <audio controls src={audioUrl} className="h-9 w-full" />
              </div>
            ) : undefined
          }
          extraMeta={[
            ...(result.duration
              ? [{ label: "Duration", value: `${result.duration.toFixed(1)}s` }]
              : []),
          ]}
        />
      )}
    </div>
  );
}
