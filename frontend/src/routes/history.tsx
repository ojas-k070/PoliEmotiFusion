import { createFileRoute } from "@tanstack/react-router";
import { Trash2 } from "lucide-react";
import { EmptyState } from "@/components/analysis/StateCards";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { clearOwnHistory, useHistory } from "@/lib/store";

export const Route = createFileRoute("/history")({ component: HistoryPage });

function HistoryPage() {
  const records = useHistory();
  return <div className="space-y-6">
    <PageHeader eyebrow="History" title="Analysis History" description="Saved analyses from this browser." actions={records.length ? <Button variant="outline" onClick={clearOwnHistory} className="gap-1.5"><Trash2 className="h-4 w-4" />Clear history</Button> : undefined} />
    {records.length === 0 ? <EmptyState title="No analyses yet" description="Run a text analysis to see saved results here." /> : <div className="space-y-3">{records.map((record) => <Card key={record.id} className="compactable gap-1 p-4"><div className="flex flex-wrap items-center justify-between gap-2"><p className="font-semibold">{record.emotion} <span className="text-sm font-normal text-muted-foreground">{record.confidence.toFixed(1)}%</span></p><p className="text-xs text-muted-foreground">{new Date(record.date).toLocaleString()}</p></div><p className="text-sm text-muted-foreground">{record.inputLabel}</p><p className="text-xs text-muted-foreground">{record.category}</p></Card>)}</div>}
  </div>;
}
