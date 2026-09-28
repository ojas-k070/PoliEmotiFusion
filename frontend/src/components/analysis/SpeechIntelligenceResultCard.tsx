import {
  Check,
  Copy,
  Download,
  FileDown,
  Globe2,
  Languages,
  RotateCcw,
  Sparkles,
  Volume2,
} from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { AudioWaveform } from "@/components/analysis/AudioWaveform";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import type { SpeechIntelligenceResult } from "@/lib/types";

interface SpeechIntelligenceResultCardProps {
  result: SpeechIntelligenceResult;
  audioUrl?: string | null;
  onAnalyzeAgain: () => void;
}

export function SpeechIntelligenceResultCard({
  result,
  audioUrl,
  onAnalyzeAgain,
}: SpeechIntelligenceResultCardProps) {
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const [copiedTranslation, setCopiedTranslation] = useState(false);
  const [playbackProgress, setPlaybackProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const copyToClipboard = async (text: string, type: "transcript" | "translation") => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === "transcript") {
        setCopiedTranscript(true);
        setTimeout(() => setCopiedTranscript(false), 2000);
        toast.success("Transcript copied to clipboard");
      } else {
        setCopiedTranslation(true);
        setTimeout(() => setCopiedTranslation(false), 2000);
        toast.success("Translation copied to clipboard");
      }
    } catch {
      toast.error("Failed to copy text to clipboard");
    }
  };

  const downloadTextFile = (content: string, filename: string) => {
    try {
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast.success(`Downloaded ${filename}`);
    } catch {
      toast.error("Failed to download text file");
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current && audioRef.current.duration) {
      setCurrentTime(audioRef.current.currentTime);
      setPlaybackProgress(audioRef.current.currentTime / audioRef.current.duration);
    }
  };

  const handleAudioEnded = () => {
    setPlaybackProgress(0);
    setCurrentTime(0);
  };

  const formatSecs = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? "0" : ""}${s}`;
  };

  const transcriptWordCount = result.transcript.trim().split(/\s+/).filter(Boolean).length;
  const translationWordCount = result.translation.trim().split(/\s+/).filter(Boolean).length;
  const baseFilename = result.inputLabel.split(" ")[0].replace(/\.[^/.]+$/, "") || "speech";

  return (
    <div className="space-y-6">
      {/* 1. Header & Speech Metadata Card */}
      <Card className="compactable space-y-6 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="gap-1.5 border-primary/40 bg-primary/10 text-primary">
                <Languages className="h-3.5 w-3.5" />
                Political Speech Intelligence
              </Badge>
              <Badge variant="secondary" className="gap-1.5 font-semibold">
                <Globe2 className="h-3.5 w-3.5 text-primary" />
                Detected Language: {result.languageName} ({result.language})
              </Badge>
              {result.chunkCount > 1 && (
                <Badge variant="outline" className="text-xs">
                  {result.chunkCount} Speech Chunks Processed
                </Badge>
              )}
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
              Multilingual Speech Intelligence & Translation
            </h2>
            <p className="text-xs text-muted-foreground">
              Processed through 16 kHz acoustic normalization and neural multilingual sequence decoding.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `speech-intelligence-${result.id}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="gap-1.5"
            >
              <Download className="h-4 w-4" />
              Export JSON
            </Button>
            <Button onClick={onAnalyzeAgain} className="gap-1.5">
              <RotateCcw className="h-4 w-4" />
              Analyze Another Audio
            </Button>
          </div>
        </div>

        <Separator />

        {/* Audio Player & Signal Telemetry */}
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="space-y-3 rounded-xl border border-border bg-surface p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <Volume2 className="h-4 w-4 text-primary" />
                <span>Audio Playback & Waveform</span>
              </div>
              <span className="font-mono text-xs text-muted-foreground">
                {formatSecs(currentTime)} / {result.formattedDuration || formatSecs(result.duration || 0)}
              </span>
            </div>

            {result.waveform && result.waveform.length > 0 && (
              <AudioWaveform data={result.waveform} progress={playbackProgress} />
            )}

            {audioUrl && (
              <div className="pt-1">
                <audio
                  ref={audioRef}
                  controls
                  src={audioUrl}
                  className="h-9 w-full"
                  onTimeUpdate={handleTimeUpdate}
                  onEnded={handleAudioEnded}
                />
              </div>
            )}
          </div>

          {/* Acoustic File Telemetry */}
          <div className="rounded-xl border border-border bg-surface p-4 space-y-3">
            <p className="text-sm font-semibold">Speech Recording Telemetry</p>
            <div className="grid grid-cols-2 gap-y-2.5 text-xs sm:grid-cols-2">
              <div>
                <span className="text-muted-foreground">Duration:</span>{" "}
                <span className="font-semibold text-foreground">{result.duration.toFixed(1)}s</span>
              </div>
              <div>
                <span className="text-muted-foreground">Sample Rate:</span>{" "}
                <span className="font-semibold text-foreground">
                  {result.audioStats?.sampleRate ? `${result.audioStats.sampleRate.toLocaleString()} Hz` : "16,000 Hz"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Channels:</span>{" "}
                <span className="font-semibold text-foreground">
                  {result.audioStats?.channelDesc || "Mono"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Audio Format:</span>{" "}
                <span className="font-semibold text-foreground">
                  {result.audioStats?.format || "WAV"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">File Size:</span>{" "}
                <span className="font-semibold text-foreground">
                  {result.audioStats?.fileSize || "N/A"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Category:</span>{" "}
                <span className="font-semibold text-foreground">{result.category}</span>
              </div>
            </div>
            <Separator />
            <div className="text-[11px] text-muted-foreground flex items-center justify-between">
              <span>Model: {result.model}</span>
              {result.processingTime ? <span>Latency: {result.processingTime}s</span> : null}
            </div>
          </div>
        </div>
      </Card>

      {/* 2. Side-by-Side Transcript & English Translation Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Original Transcript Card */}
        <Card className="compactable flex flex-col justify-between p-6 space-y-4">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-foreground flex items-center gap-2">
                  <span>Original Transcript</span>
                  <Badge variant="secondary" className="text-[11px]">
                    {result.languageName}
                  </Badge>
                </h3>
                <p className="text-xs text-muted-foreground">
                  Verbatim speech decoded in native script
                </p>
              </div>

              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(result.transcript, "transcript")}
                  className="h-8 gap-1 px-2.5 text-xs"
                >
                  {copiedTranscript ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                  {copiedTranscript ? "Copied" : "Copy Transcript"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => downloadTextFile(result.transcript, `${baseFilename}-transcript-${result.language}.txt`)}
                  className="h-8 gap-1 px-2.5 text-xs"
                  title="Download Transcript (.txt)"
                >
                  <FileDown className="h-3.5 w-3.5" />
                  Download (.txt)
                </Button>
              </div>
            </div>

            <Separator />

            {/* Transcript Display Box */}
            <div className="min-h-[220px] max-h-[460px] overflow-y-auto rounded-xl border border-border/80 bg-background/60 p-4 text-sm leading-relaxed text-foreground select-text font-normal shadow-inner">
              <p className="whitespace-pre-wrap">{result.transcript}</p>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border/60">
            <span>{transcriptWordCount} words · {result.transcript.length} characters</span>
            <span>Script: {result.language === "hi" || result.language === "mr" ? "Devanagari Unicode" : "Standard Latin"}</span>
          </div>
        </Card>

        {/* English Translation Card */}
        <Card className="compactable flex flex-col justify-between p-6 space-y-4 border-primary/20 bg-primary/[0.02]">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-foreground flex items-center gap-2">
                  <span>English Translation</span>
                  <Badge variant="outline" className="border-primary/40 text-primary text-[11px] gap-1">
                    <Sparkles className="h-3 w-3" />
                    Fluent Translation
                  </Badge>
                </h3>
                <p className="text-xs text-muted-foreground">
                  Preserves rhetorical style, tone, context, and speaker intent
                </p>
              </div>

              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(result.translation, "translation")}
                  className="h-8 gap-1 px-2.5 text-xs"
                >
                  {copiedTranslation ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                  {copiedTranslation ? "Copied" : "Copy Translation"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => downloadTextFile(result.translation, `${baseFilename}-translation-en.txt`)}
                  className="h-8 gap-1 px-2.5 text-xs"
                  title="Download Translation (.txt)"
                >
                  <FileDown className="h-3.5 w-3.5" />
                  Download (.txt)
                </Button>
              </div>
            </div>

            <Separator />

            {/* Translation Display Box */}
            <div className="min-h-[220px] max-h-[460px] overflow-y-auto rounded-xl border border-primary/20 bg-background/80 p-4 text-sm leading-relaxed text-foreground select-text font-normal shadow-inner">
              <p className="whitespace-pre-wrap">{result.translation}</p>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-border/60">
            <span>{translationWordCount} words · {result.translation.length} characters</span>
            <span>
              {result.isTranslationNeeded ? `Translated from ${result.languageName}` : "Native English Speech"}
            </span>
          </div>
        </Card>
      </div>

      {/* 3. Action Bar */}
      <div className="flex justify-end">
        <Button onClick={onAnalyzeAgain} size="lg" className="gap-2 shadow-sm">
          <RotateCcw className="h-4 w-4" />
          Analyze Another Audio
        </Button>
      </div>
    </div>
  );
}
