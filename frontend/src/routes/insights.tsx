import { createFileRoute } from "@tanstack/react-router";
import { EmptyState } from "@/components/analysis/StateCards";
import { PageHeader } from "@/components/common/PageHeader";
import { Card } from "@/components/ui/card";
import { EMOTIONS } from "@/lib/types";
import { useHistory } from "@/lib/store";

export const Route = createFileRoute("/insights")({ component: InsightsPage });

function InsightsPage() {
  const records = useHistory();
  const counts = Object.fromEntries(EMOTIONS.map((emotion) => [emotion, 0])) as Record<(typeof EMOTIONS)[number], number>;
  records.forEach((record) => { counts[record.emotion] += 1; });
  const maximum = Math.max(1, ...Object.values(counts));
  return <div className="space-y-6"><PageHeader eyebrow="Insights" title="Emotion Insights" description="Distribution calculated from saved analyses." />
    {records.length === 0 ? <EmptyState title="No insights available" description="Insights will appear after you run text analyses." /> : <Card className="compactable gap-5 p-6"><h2 className="font-semibold">Emotion distribution</h2><div className="space-y-3">{EMOTIONS.map((emotion) => <div key={emotion} className="grid grid-cols-[90px_1fr_36px] items-center gap-3 text-sm"><span>{emotion}</span><div className="h-2 overflow-hidden rounded bg-secondary"><div className="h-full rounded bg-primary" style={{ width: `${(counts[emotion] / maximum) * 100}%` }} /></div><span className="text-right text-muted-foreground">{counts[emotion]}</span></div>)}</div></Card>}
  </div>;
}
