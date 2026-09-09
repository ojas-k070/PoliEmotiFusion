import { CloudUpload, FileCheck2, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { Button } from "@/components/ui/button";
import { formatBytes } from "@/lib/emotions";
import { FILE_RULES } from "@/lib/validation";
import type { Modality } from "@/lib/types";
import { cn } from "@/lib/utils";

export function UploadZone({
  modality,
  file,
  onSelect,
  onClear,
}: {
  modality: Exclude<Modality, "text">;
  file: File | null;
  onSelect: (file: File) => void;
  onClear: () => void;
}) {
  const rules = FILE_RULES[modality];
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onSelect(dropped);
  };

  if (file) {
    return (
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-secondary/50 p-4">
        <div className="flex min-w-0 items-center gap-3">
          <span className="rounded-lg bg-primary/10 p-2 text-primary">
            <FileCheck2 className="h-5 w-5" />
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">{file.name}</p>
            <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
          </div>
        </div>
        <Button variant="ghost" size="sm" onClick={onClear} className="gap-1.5">
          <X className="h-4 w-4" />
          Remove
        </Button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-border bg-surface px-6 py-12 text-center transition-colors",
        dragging && "border-primary bg-primary/5",
      )}
    >
      <span className="rounded-xl bg-secondary p-3 text-primary">
        <CloudUpload className="h-6 w-6" />
      </span>
      <div className="space-y-1">
        <p className="text-sm font-semibold">Drag & drop your {modality} file here</p>
        <p className="text-xs text-muted-foreground">or select a file from your device</p>
      </div>
      <Button variant="outline" onClick={() => inputRef.current?.click()}>
        Browse files
      </Button>
      <p className="text-xs text-muted-foreground">
        Supported: {rules.extensions.join(", ").toUpperCase()} · Max{" "}
        {Math.round(rules.maxBytes / (1024 * 1024))} MB
        {rules.maxDuration ? ` · Max ${rules.maxDuration / 60} min` : ""}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={rules.accept}
        className="hidden"
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (selected) onSelect(selected);
          e.target.value = "";
        }}
      />
    </div>
  );
}
