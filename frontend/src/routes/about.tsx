import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/common/PageHeader";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/about")({ component: AboutPage });

function AboutPage() {
  return <div className="space-y-6">
    <PageHeader eyebrow="About" title="PoliEmotiFusion" description="Multimodal emotion analysis for political discourse." />
    <Card className="compactable gap-3 p-6"><h2 className="text-lg font-bold">Purpose</h2><p className="text-sm leading-relaxed text-muted-foreground">The application studies emotion signals in political text, images, audio, and video. Text analysis currently uses a DistilRoBERTa emotion classifier.</p></Card>
    <Card className="compactable gap-3 p-6"><h2 className="text-lg font-bold">Responsible use</h2><p className="text-sm leading-relaxed text-muted-foreground">Predictions are model outputs. They do not establish ideology, political preference, intent, truthfulness, or a person’s internal state.</p></Card>
  </div>;
}
