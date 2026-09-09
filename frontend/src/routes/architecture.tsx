import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/common/PageHeader";
import { Card } from "@/components/ui/card";

export const Route = createFileRoute("/architecture")({ component: ArchitecturePage });

function ArchitecturePage() {
  return <div className="space-y-6">
    <PageHeader eyebrow="Architecture" title="Multimodal emotion analysis" description="Each modality uses a dedicated model pipeline and returns a common emotion taxonomy." />
    <div className="grid gap-4 md:grid-cols-2">
      <Card className="compactable gap-2 p-5"><h2 className="font-bold">Text</h2><p className="text-sm text-muted-foreground">FastAPI → tokenizer → DistilRoBERTa classifier → seven emotion probabilities.</p></Card>
      <Card className="compactable gap-2 p-5"><h2 className="font-bold">Image</h2><p className="text-sm text-muted-foreground">Reserved for a facial-expression and visual emotion pipeline.</p></Card>
      <Card className="compactable gap-2 p-5"><h2 className="font-bold">Audio</h2><p className="text-sm text-muted-foreground">Reserved for a speech-emotion pipeline based on acoustic features.</p></Card>
      <Card className="compactable gap-2 p-5"><h2 className="font-bold">Video</h2><p className="text-sm text-muted-foreground">Reserved for temporal aggregation of visual and audio emotion signals.</p></Card>
    </div>
  </div>;
}
