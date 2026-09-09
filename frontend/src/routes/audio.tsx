import { createFileRoute } from "@tanstack/react-router";
import { Mic } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { CategorySelect } from "@/components/analysis/CategorySelect";
import { UploadZone } from "@/components/analysis/UploadZone";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { PoliticalCategory } from "@/lib/types";
import { validateFile } from "@/lib/validation";

export const Route = createFileRoute("/audio")({ component: AudioAnalysisPage });

function AudioAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState<PoliticalCategory>("Political Speech");
  const [error, setError] = useState<string | null>(null);
  const select = (selected: File) => {
    const validation = validateFile("audio", selected);
    if (validation) { setError(validation); toast.error(validation); return; }
    setError(null); setFile(selected);
  };
  const clear = () => { setFile(null); setError(null); };
  return <div className="space-y-6">
    <PageHeader eyebrow="Speech analysis" title="Political Audio Emotion Analysis" description="Upload a political speech or interview recording." />
    <Card className="compactable gap-5 p-6">
      <UploadZone modality="audio" file={file} onSelect={select} onClear={clear} />
      {error && <p className="text-xs font-medium text-destructive">{error}</p>}
      <CategorySelect value={category} onChange={setCategory} />
      <p className="rounded-lg border border-border bg-secondary/40 px-3.5 py-2.5 text-xs text-muted-foreground">This analysis module is not available yet.</p>
      <div className="flex gap-2"><Button disabled className="gap-1.5"><Mic className="h-4 w-4" />Analyze Audio</Button><Button variant="outline" onClick={clear} disabled={!file}>Clear</Button></div>
    </Card>
  </div>;
}
