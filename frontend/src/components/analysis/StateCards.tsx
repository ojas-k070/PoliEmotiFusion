import { AlertTriangle, Loader2, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { LOADING_MESSAGES } from "@/services/config";
import type { Modality } from "@/lib/types";

export function LoadingState({ modality }: { modality: Modality }) {
  const steps = LOADING_MESSAGES[modality];
  const [progress, setProgress] = useState(8);

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((p) => (p >= 95 ? 95 : p + Math.random() * 12));
    }, 320);
    return () => clearInterval(timer);
  }, []);

  const stepIndex = Math.min(steps.length - 1, Math.floor((progress / 100) * steps.length));

  return (
    <Card className="compactable gap-4 p-6">
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
        <div>
          <p className="text-sm font-semibold">Analyzing political content...</p>
          <p className="text-xs text-muted-foreground">{steps[stepIndex]}</p>
        </div>
      </div>
      <Progress value={progress} className="h-2" />
      <div className="grid gap-3 sm:grid-cols-3">
        <Skeleton className="h-20 rounded-xl" />
        <Skeleton className="h-20 rounded-xl" />
        <Skeleton className="h-20 rounded-xl" />
      </div>
      <Skeleton className="h-40 rounded-xl" />
    </Card>
  );
}

export function ErrorState({
  message,
  onRetry,
  onReset,
  resetLabel = "Upload another file",
}: {
  message?: string;
  onRetry: () => void;
  onReset: () => void;
  resetLabel?: string;
}) {
  return (
    <Card className="compactable gap-4 border-destructive/30 bg-destructive/5 p-6">
      <div className="flex items-start gap-3">
        <span className="rounded-lg bg-destructive/10 p-2 text-destructive">
          <AlertTriangle className="h-5 w-5" />
        </span>
        <div>
          <p className="text-sm font-semibold">Analysis could not be completed.</p>
          <p className="mt-1 text-sm text-muted-foreground">
            {message ?? "Something interrupted the request. Please try again."}
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button onClick={onRetry}>Try again</Button>
        <Button variant="outline" onClick={onReset}>
          {resetLabel}
        </Button>
      </div>
    </Card>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <Card className="compactable items-center gap-3 border-dashed p-10 text-center">
      <span className="rounded-xl bg-secondary p-3 text-primary">
        <Sparkles className="h-5 w-5" />
      </span>
      <p className="text-sm font-semibold">{title}</p>
      <p className="max-w-md text-sm text-muted-foreground">{description}</p>
      {action && (
        <Button variant="outline" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </Card>
  );
}
