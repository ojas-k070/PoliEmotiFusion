import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

export function PipelineRow({
  title,
  steps,
  badges = ["Independent Pipeline", "Future Backend Integration"],
}: {
  title: string;
  steps: string[];
  badges?: string[];
}) {
  return (
    <Card className="compactable gap-3 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-bold">{title}</p>
        <div className="flex flex-wrap gap-1.5">
          {badges.map((b) => (
            <Badge key={b} variant="outline" className="text-[10px] uppercase tracking-wide">
              {b}
            </Badge>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {steps.map((step, i) => (
          <div key={step} className="flex items-center gap-2">
            <span className="rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-medium">
              {step}
            </span>
            {i < steps.length - 1 && <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />}
          </div>
        ))}
      </div>
    </Card>
  );
}

export function SystemFlowDiagram() {
  const nodes = [
    "User",
    "Input validation",
    "Modality router",
    "Text / Image / Video / Audio",
    "Independent AI model",
    "Emotion + probability + confidence",
    "Result dashboard",
  ];
  return (
    <div className="rounded-2xl bg-hero-navy p-6 text-navy-foreground">
      <p className="text-xs font-semibold uppercase tracking-wider text-navy-foreground/70">
        System flow
      </p>
      <div className="mt-4 flex flex-col items-stretch gap-2">
        {nodes.map((node, i) => (
          <div key={node} className="flex flex-col items-center gap-2">
            <div className="w-full rounded-xl border border-white/15 bg-white/10 px-4 py-2.5 text-center text-sm font-medium">
              {node}
            </div>
            {i < nodes.length - 1 && <span className="text-navy-foreground/50">↓</span>}
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs leading-relaxed text-navy-foreground/70">
        No multimodal fusion is performed. Each modality is independently selectable and
        independently processed.
      </p>
    </div>
  );
}
