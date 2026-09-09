import { Download, RotateCcw } from "lucide-react";
import type { ReactNode } from "react";
import { ProbabilityChart } from "@/components/analysis/charts";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { EMOTION_TEXT_CLASS, formatConfidence } from "@/lib/emotions";
import type { AnalysisResult } from "@/lib/types";

function MetaRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

export function downloadResult(result: AnalysisResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `poliemotifusion-${result.modality}-${result.id}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function ResultCard({
  result,
  onAnalyzeAgain,
  preview,
  extras,
  extraMeta,
}: {
  result: AnalysisResult;
  onAnalyzeAgain: () => void;
  /** modality-specific preview (image, video player, waveform) */
  preview?: ReactNode;
  /** modality-specific panels rendered below the main grid */
  extras?: ReactNode;
  extraMeta?: { label: string; value: string | number }[];
}) {
  return (
    <div className="space-y-5">
      <Card className="compactable gap-5 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Prediction
            </p>
            <div className="flex items-end gap-3">
              <h2 className={`text-3xl font-bold ${EMOTION_TEXT_CLASS[result.emotion]}`}>
                {result.emotion}
              </h2>
              <span className="pb-1 text-sm text-muted-foreground">
                Confidence {formatConfidence(result.confidence)}
              </span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => downloadResult(result)} className="gap-1.5">
              <Download className="h-4 w-4" />
              Export result
            </Button>
            <Button onClick={onAnalyzeAgain} className="gap-1.5">
              <RotateCcw className="h-4 w-4" />
              Analyze again
            </Button>
          </div>
        </div>

        <Separator />

        <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
          <div className="space-y-3">
            <p className="text-sm font-semibold">Probability distribution</p>
            <ProbabilityChart probabilities={result.probabilities} />
          </div>
          <div className="space-y-4">
            {preview}
            <div className="rounded-xl border border-border bg-surface p-4">
              <p className="mb-1 text-sm font-semibold">Analysis details</p>
              <MetaRow label="Modality" value={result.modality.toUpperCase()} />
              <MetaRow label="Category" value={result.category} />
              <MetaRow label="Model pipeline" value={result.model} />
              <MetaRow label="Timestamp" value={new Date(result.timestamp).toLocaleString()} />
              <MetaRow label="Input" value={result.inputLabel} />
              {extraMeta?.map((m) => <MetaRow key={m.label} label={m.label} value={m.value} />)}
            </div>
          </div>
        </div>
      </Card>

      {extras}

      <Card className="compactable gap-2 p-6">
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold">Analysis summary</p>
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">{result.summary}</p>
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Emotion predictions represent model outputs and should not be interpreted as statements of
          political ideology, intent, truthfulness, or political preference.
        </p>
      </Card>
    </div>
  );
}
