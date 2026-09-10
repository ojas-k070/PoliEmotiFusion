import { createFileRoute } from "@tanstack/react-router";
import {
  ScanFace,
  Loader2,
  CheckCircle2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { CategorySelect } from "@/components/analysis/CategorySelect";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

import type {
  AnalysisResult,
  PoliticalCategory,
} from "@/lib/types";

import { validateFile } from "@/lib/validation";

import { analyzeImage } from "@/services/analysisService";

export const Route = createFileRoute("/image")({
  head: () => ({
    meta: [
      {
        title:
          "Political Image Emotion Analysis — PoliEmotiFusion",
      },
      {
        name: "description",
        content:
          "Upload a political image to analyze emotional signals with confidence scores.",
      },
      {
        property: "og:title",
        content:
          "Political Image Emotion Analysis — PoliEmotiFusion",
      },
      {
        property: "og:description",
        content:
          "Emotion signals from political imagery, with probability distribution.",
      },
    ],
  }),

  component: ImageAnalysisPage,
});

function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [category, setCategory] =
    useState<PoliticalCategory>("Political News");

  const [error, setError] =
    useState<string | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [result, setResult] =
    useState<AnalysisResult | null>(null);

  /*
   * Create image preview when a file is selected.
   */
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);

    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  /*
   * Handle image selection.
   */
  const select = (selected: File) => {
    const validation =
      validateFile("image", selected);

    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }

    setError(null);
    setResult(null);
    setFile(selected);
  };

  /*
   * Clear selected image and result.
   */
  const clear = () => {
    setFile(null);
    setError(null);
    setResult(null);
  };

  /*
   * Send image to FastAPI backend.
   */
  const handleAnalyze = async () => {
    if (!file) {
      toast.error(
        "Please select an image first.",
      );
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const analysis =
        await analyzeImage({
          file,
          category,
        });

      setResult(analysis);

      toast.success(
        "Image analysis completed.",
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to analyze image.";

      setError(message);

      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Vision pipeline · Image Emotion"
        title="Political Image Emotion Analysis"
        description="Upload an image containing a political figure or political context to analyze emotional signals."
      />

      <Card className="compactable gap-5 p-6">

        {/* Upload area */}
        <UploadZone
          modality="image"
          file={file}
          onSelect={select}
          onClear={clear}
        />

        {/* Error message */}
        {error && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3">
            <p className="text-sm font-medium text-destructive">
              {error}
            </p>
          </div>
        )}

        {/* Image preview */}
        {previewUrl && (
          <div className="overflow-hidden rounded-xl border border-border">
            <img
              src={previewUrl}
              alt="Selected image preview"
              className="max-h-80 w-full object-contain bg-secondary/40"
            />
          </div>
        )}

        {/* Political category */}
        <CategorySelect
          value={category}
          onChange={(value) => {
            setCategory(value);
            setResult(null);
          }}
        />

        {/* Information message */}
        {!result && (
          <div className="rounded-lg border border-border bg-secondary/40 px-3.5 py-2.5 text-xs text-muted-foreground">
            Upload a political image and click
            Analyze Image to run EfficientNet-B2
            emotion analysis.
          </div>
        )}

        {/* Buttons */}
        <div className="flex flex-wrap gap-2">

          <Button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="gap-1.5"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <ScanFace className="h-4 w-4" />
                Analyze Image
              </>
            )}
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

      {/* Analysis result */}
      {result && (
        <ImageResults result={result} />
      )}
    </div>
  );
}

/**
 * Display image emotion analysis result.
 */
function ImageResults({
  result,
}: {
  result: AnalysisResult;
}) {
  const probabilities =
    result.probabilities ?? {};

  const sortedEmotions =
    Object.entries(probabilities).sort(
      ([, a], [, b]) => b - a,
    );

  return (
    <Card className="space-y-6 p-6">

      {/* Result heading */}
      <div className="flex items-center gap-3">
        <div className="rounded-full bg-primary/10 p-2">
          <CheckCircle2 className="h-5 w-5 text-primary" />
        </div>

        <div>
          <h2 className="text-lg font-semibold">
            Image Analysis Result
          </h2>

          <p className="text-sm text-muted-foreground">
            Analysis completed using{" "}
            {result.model}
          </p>
        </div>
      </div>

      {/* Dominant emotion */}
      <div className="rounded-xl border border-border bg-secondary/40 p-5">

        <p className="text-sm text-muted-foreground">
          Dominant Emotion
        </p>

        <div className="mt-1 flex items-baseline gap-3">

          <h3 className="text-3xl font-bold">
            {result.emotion}
          </h3>

          <span className="text-lg font-semibold text-muted-foreground">
            {Number(result.confidence).toFixed(1)}%
          </span>

        </div>

        {result.summary && (
          <p className="mt-2 text-sm text-muted-foreground">
            {result.summary}
          </p>
        )}

      </div>

      {/* Probability distribution */}
      <div>

        <h3 className="mb-4 text-base font-semibold">
          Emotion Probability Distribution
        </h3>

        <div className="space-y-4">

          {sortedEmotions.map(
            ([emotion, probability]) => (
              <div key={emotion}>

                <div className="mb-1.5 flex items-center justify-between">

                  <span className="text-sm font-medium">
                    {emotion}
                  </span>

                  <span className="text-sm text-muted-foreground">
                    {Number(probability).toFixed(1)}%
                  </span>

                </div>

                <div className="h-2 overflow-hidden rounded-full bg-secondary">

                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{
                      width: `${Math.min(
                        Math.max(
                          Number(probability),
                          0,
                        ),
                        100,
                      )}%`,
                    }}
                  />

                </div>

              </div>
            ),
          )}

        </div>
      </div>

      {/* Metadata */}
      <div className="grid gap-3 border-t border-border pt-5 sm:grid-cols-2 lg:grid-cols-4">

        <MetadataItem
          label="Category"
          value={result.category}
        />

        <MetadataItem
          label="Model"
          value={result.model}
        />

        <MetadataItem
          label="Input"
          value={result.inputLabel}
        />

        <MetadataItem
          label="Timestamp"
          value={new Date(
            result.timestamp,
          ).toLocaleString()}
        />

      </div>
    </Card>
  );
}

/**
 * Small metadata card.
 */
function MetadataItem({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg bg-secondary/40 p-3">

      <p className="text-xs text-muted-foreground">
        {label}
      </p>

      <p className="mt-1 break-words text-sm font-medium">
        {value}
      </p>

    </div>
  );
}