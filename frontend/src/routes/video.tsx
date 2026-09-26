import { createFileRoute } from "@tanstack/react-router";
import { Video } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { AnalysisResult, PoliticalCategory } from "@/lib/types";
import { validateFile } from "@/lib/validation";
import { analyzeVideo } from "@/services/analysisService";

export const Route = createFileRoute("/video")({
  component: VideoAnalysisPage,
});

function VideoAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] =
    useState<PoliticalCategory>("Political News");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const select = (selected: File) => {
    const validation = validateFile("video", selected);

    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }

    setError(null);
    setResult(null);
    setFile(selected);
  };

  const clear = () => {
    setFile(null);
    setError(null);
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!file) {
      const message = "Please select a video first.";
      setError(message);
      toast.error(message);
      return;
    }

    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const analysisResult = await analyzeVideo({
        file,
        category,
      });

      setResult(analysisResult);
      toast.success("Video analysis completed.");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Video analysis failed.";

      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Video analysis"
        title="Political Video Emotion Analysis"
        description="Upload a political video for temporal emotion analysis."
      />

      <Card className="compactable gap-5 p-6">
        <UploadZone
          modality="video"
          file={file}
          onSelect={select}
          onClear={clear}
        />

        {error && (
          <p className="text-xs font-medium text-destructive">{error}</p>
        )}

        <CategorySelect value={category} onChange={setCategory} />

        <p className="rounded-lg border border-border bg-secondary/40 px-3.5 py-2.5 text-xs text-muted-foreground">
          The video is analyzed across temporal windows using a pretrained
          facial-expression recognition model.
        </p>

        <div className="flex gap-2">
          <Button
            disabled={!file || loading}
            onClick={handleAnalyze}
            className="gap-1.5"
          >
            <Video className="h-4 w-4" />
            {loading ? "Analyzing..." : "Analyze Video"}
          </Button>

          <Button
            variant="outline"
            onClick={clear}
            disabled={!file || loading}
          >
            Clear
          </Button>
        </div>
      </Card>

      {result && (
        <Card className="gap-4 p-6">
          <div>
            <h2 className="text-lg font-semibold">Analysis Result</h2>
            <p className="text-sm text-muted-foreground">
              {result.inputLabel}
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-border p-4">
              <p className="text-xs text-muted-foreground">
                Dominant Emotion
              </p>
              <p className="mt-1 text-xl font-semibold">{result.emotion}</p>
            </div>

            <div className="rounded-lg border border-border p-4">
              <p className="text-xs text-muted-foreground">Confidence</p>
              <p className="mt-1 text-xl font-semibold">
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="rounded-lg border border-border p-4">
              <p className="text-xs text-muted-foreground">Frames Processed</p>
              <p className="mt-1 text-xl font-semibold">
                {result.frames ?? 0}
              </p>
            </div>

            <div className="rounded-lg border border-border p-4">
              <p className="text-xs text-muted-foreground">Duration</p>
              <p className="mt-1 text-xl font-semibold">
                {(result.duration ?? 0).toFixed(1)}s
              </p>
            </div>
          </div>

          <div>
            <h3 className="mb-2 text-sm font-semibold">Emotion Probabilities</h3>

            <div className="space-y-2">
              {Object.entries(result.probabilities).map(
                ([emotion, probability]) => (
                  <div
                    key={emotion}
                    className="flex items-center justify-between rounded-lg border border-border px-3 py-2"
                  >
                    <span className="text-sm">{emotion}</span>
                    <span className="text-sm font-medium">
                      {(probability * 100).toFixed(1)}%
                    </span>
                  </div>
                ),
              )}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-secondary/40 p-4">
            <p className="text-sm text-muted-foreground">{result.summary}</p>
          </div>
        </Card>
      )}
    </div>
  );
}