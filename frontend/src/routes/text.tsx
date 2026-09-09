import { createFileRoute } from "@tanstack/react-router";
import { Sparkles } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { ResultCard } from "@/components/analysis/ResultCard";
import { ErrorState, LoadingState } from "@/components/analysis/StateCards";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { addAnalysis } from "@/lib/store";
import type { AnalysisResult, PoliticalCategory } from "@/lib/types";
import { TEXT_LIMITS, validateText } from "@/lib/validation";
import { analyzeText } from "@/services/analysisService";
import { TEXT_EXAMPLES } from "@/services/config";

export const Route = createFileRoute("/text")({
  head: () => ({
    meta: [
      { title: "Political Text Emotion Analysis — PoliEmotiFusion" },
      {
        name: "description",
        content:
          "Analyze emotions in political speeches, debate statements, articles and social-media posts.",
      },
      { property: "og:title", content: "Political Text Emotion Analysis — PoliEmotiFusion" },
      {
        property: "og:description",
        content: "Submit political text and review emotion probabilities and confidence.",
      },
    ],
  }),
  component: TextAnalysisPage,
});

type Status = "idle" | "loading" | "success" | "error";

function TextAnalysisPage() {
  const [text, setText] = useState("");
  const [category, setCategory] = useState<PoliticalCategory>("Political Speech");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const run = async () => {
    const validation = validateText(text);
    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }
    setError(null);
    setStatus("loading");
    try {
      const res = await analyzeText({ text, category });
      setResult(res);
      addAnalysis(res);
      setStatus("success");
      toast.success(`Detected emotion: ${res.emotion}`);
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Text analysis failed.";
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
        eyebrow="Text pipeline · DistilRoBERTa Emotion"
        title="Political Text Emotion Analysis"
        description="Enter political text such as a speech excerpt, political statement, debate statement, article excerpt, or social-media post."
      />

      <Card className="compactable gap-5 p-6">
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {TEXT_EXAMPLES.map((ex) => (
              <Button
                key={ex.label}
                variant="secondary"
                size="sm"
                onClick={() => {
                  setText(ex.text);
                  setCategory(ex.category);
                  setError(null);
                }}
              >
                {ex.label}
              </Button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Example inputs for quick testing.
          </p>
        </div>

        <div className="space-y-2">
          <Textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              if (error) setError(null);
            }}
            rows={9}
            maxLength={TEXT_LIMITS.max}
            placeholder="Paste a political statement, speech excerpt, debate statement, article text, or social-media post..."
            className="resize-y text-sm leading-relaxed"
            aria-invalid={Boolean(error)}
          />
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <span className={error ? "font-medium text-destructive" : "text-muted-foreground"}>
              {error ?? `Minimum ${TEXT_LIMITS.min} characters`}
            </span>
            <span className="text-muted-foreground tabular-nums">
              {text.length} / {TEXT_LIMITS.max}
            </span>
          </div>
        </div>

        <CategorySelect value={category} onChange={setCategory} />

        <div className="flex flex-wrap gap-2">
          <Button onClick={run} disabled={status === "loading"} className="gap-1.5">
            <Sparkles className="h-4 w-4" />
            {status === "loading" ? "Analyzing..." : "Analyze Emotion of my text"}
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              setText("");
              reset();
            }}
            disabled={status === "loading"}
          >
            Clear
          </Button>
        </div>
      </Card>

      {status === "loading" && <LoadingState modality="text" />}
      {status === "error" && <ErrorState onRetry={run} onReset={reset} resetLabel="Edit input" />}
      {status === "success" && result && <ResultCard result={result} onAnalyzeAgain={reset} />}
    </div>
  );
}
