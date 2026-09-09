import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import { PageHeader } from "@/components/common/PageHeader";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { clearOwnHistory, useSettings, type ThemeMode } from "@/lib/store";
import type { Modality } from "@/lib/types";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — PoliEmotiFusion" },
      {
        name: "description",
        content:
          "Choose appearance, motion, density, notification and default modality preferences for the analysis workspace.",
      },
      { property: "og:title", content: "Settings — PoliEmotiFusion" },
      {
        property: "og:description",
        content: "Appearance, motion, density and default modality preferences.",
      },
    ],
  }),
  component: SettingsPage,
});

const THEMES: { value: ThemeMode; label: string }[] = [
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
  { value: "system", label: "Match system" },
];

const MODALITIES: { value: Modality; label: string }[] = [
  { value: "text", label: "Text" },
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
];

function Row({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border py-4 last:border-0">
      <div className="max-w-xl space-y-1">
        <p className="text-sm font-semibold">{title}</p>
        <p className="text-sm leading-relaxed text-muted-foreground">{description}</p>
      </div>
      {children}
    </div>
  );
}

function SettingsPage() {
  const [settings, update] = useSettings();

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Settings"
        title="Workspace preferences"
        description="These preferences are stored in your browser only. Nothing is sent anywhere."
      />

      <Card className="compactable gap-0 p-6">
        <Row title="Appearance" description="Choose a light or dark workspace, or follow your device setting.">
          <div className="w-[190px]">
            <Label className="sr-only" htmlFor="theme">
              Appearance
            </Label>
            <Select value={settings.theme} onValueChange={(v) => update({ theme: v as ThemeMode })}>
              <SelectTrigger id="theme">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {THEMES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </Row>

        <Row title="Animations" description="Turn off motion and transitions for a calmer, faster interface.">
          <Switch
            checked={settings.animations}
            onCheckedChange={(v) => update({ animations: v })}
            aria-label="Animations"
          />
        </Row>

        <Row title="Compact layout" description="Reduce padding so more information fits on screen.">
          <Switch
            checked={settings.compact}
            onCheckedChange={(v) => update({ compact: v })}
            aria-label="Compact layout"
          />
        </Row>

        <Row title="Notifications" description="Show a confirmation message when an analysis finishes.">
          <Switch
            checked={settings.notifications}
            onCheckedChange={(v) => update({ notifications: v })}
            aria-label="Notifications"
          />
        </Row>

        <Row title="Default analysis type" description="The modality selected first when you start a new analysis.">
          <div className="w-[190px]">
            <Select
              value={settings.defaultModality}
              onValueChange={(v) => update({ defaultModality: v as Modality })}
            >
              <SelectTrigger aria-label="Default analysis type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODALITIES.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </Row>
      </Card>

      <Card className="compactable gap-3 p-6">
        <h2 className="text-lg font-bold">Stored data</h2>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Your analyses are kept in this browser so history survives a refresh. Clearing removes
          all locally saved analysis records.
        </p>
        <div>
          <Button
            variant="outline"
            onClick={() => {
              clearOwnHistory();
              toast.success("Your saved analyses were cleared.");
            }}
          >
            Clear saved analyses
          </Button>
        </div>
      </Card>
    </div>
  );
}
