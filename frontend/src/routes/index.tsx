import { createFileRoute } from "@tanstack/react-router";
import { BarChart3, FileText, Target } from "lucide-react";
import { EmptyState } from "@/components/analysis/StateCards";
import { PageHeader } from "@/components/common/PageHeader";
import { StatCard } from "@/components/common/StatCard";
import { Card } from "@/components/ui/card";
import { useHistory } from "@/lib/store";

export const Route = createFileRoute("/")({ component: DashboardPage });

function DashboardPage() {
  const records = useHistory();
  const confidenceRecords = records.filter((record) => record.modality !== "image");
  const average = confidenceRecords.length ? confidenceRecords.reduce((total, record) => total + record.confidence, 0) / confidenceRecords.length : null;
  const mostDetected = records.length ? Object.entries(records.reduce<Record<string, number>>((all, record) => ({ ...all, [record.emotion]: (all[record.emotion] ?? 0) + 1 }), {})).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—" : "—";
  return <div className="space-y-6"><PageHeader eyebrow="Overview" title="Political Emotion Intelligence" description="Review analyses saved in this browser." />
    <div className="grid gap-4 sm:grid-cols-3"><StatCard label="Analyses" value={records.length} icon={FileText} /><StatCard label="Average confidence" value={average === null ? "—" : `${average.toFixed(1)}%`} icon={Target} /><StatCard label="Most detected" value={mostDetected} icon={BarChart3} /></div>
    {records.length === 0 ? <EmptyState title="Start with text analysis" description="Submit political text to generate your first analysis." /> : <Card className="compactable gap-3 p-6"><h2 className="font-semibold">Recent analyses</h2>{records.slice(0, 5).map((record) => <div key={record.id} className="flex items-center justify-between border-b border-border py-2 text-sm last:border-0"><span>{record.inputLabel}</span><span className="font-medium">{record.emotion}</span></div>)}</Card>}
  </div>;
}
