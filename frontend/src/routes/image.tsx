import { createFileRoute } from "@tanstack/react-router";
import { ScanFace } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { PoliticalCategory } from "@/lib/types";
import { validateFile } from "@/lib/validation";

export const Route = createFileRoute("/image")({
  head: () => ({
    meta: [
      { title: "Political Image Emotion Analysis — PoliEmotiFusion" },
      {
        name: "description",
        content:
          "Upload a political image to analyze facial emotional signals with confidence scores.",
      },
      { property: "og:title", content: "Political Image Emotion Analysis — PoliEmotiFusion" },
      {
        property: "og:description",
        content: "Facial emotion signals from political imagery, with probability distribution.",
      },
    ],
  }),
  component: ImageAnalysisPage,
});

function ImageAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [category, setCategory] = useState<PoliticalCategory>("Political News");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const select = (selected: File) => {
    const validation = validateFile("image", selected);
    if (validation) {
      setError(validation);
      toast.error(validation);
      return;
    }
    setError(null);
    setFile(selected);
  };

  const clear = () => {
    setFile(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Vision pipeline · Image Emotion"
        title="Political Image Emotion Analysis"
        description="Upload an image containing a political figure or political context to analyze facial emotional signals."
      />

      <Card className="compactable gap-5 p-6">
        <UploadZone modality="image" file={file} onSelect={select} onClear={clear} />
        {error && <p className="text-xs font-medium text-destructive">{error}</p>}

        {previewUrl && (
          <div className="overflow-hidden rounded-xl border border-border">
            <img src={previewUrl} alt="Selected image preview" className="max-h-80 w-full object-contain bg-secondary/40" />
          </div>
        )}

        <CategorySelect value={category} onChange={setCategory} />

        <div className="rounded-lg border border-border bg-secondary/40 px-3.5 py-2.5 text-xs text-muted-foreground">
          This analysis module is not available yet.
        </div>

        <div className="flex flex-wrap gap-2">
          <Button disabled className="gap-1.5">
            <ScanFace className="h-4 w-4" />
            Analyze Image
          </Button>
          <Button variant="outline" onClick={clear} disabled={!file}>
            Clear
          </Button>
        </div>
      </Card>
    </div>
  );
}
