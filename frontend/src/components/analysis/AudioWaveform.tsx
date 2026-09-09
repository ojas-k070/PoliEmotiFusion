export function AudioWaveform({
  data,
  progress = 0,
}: {
  data: number[];
  progress?: number;
}) {
  return (
    <div className="flex h-24 items-end gap-[3px] overflow-hidden rounded-xl bg-secondary/70 p-3">
      {data.map((v, i) => {
        const played = i / data.length <= progress;
        return (
          <span
            key={i}
            className="flex-1 rounded-full transition-colors"
            style={{
              height: `${Math.max(6, v * 100)}%`,
              backgroundColor: played ? "var(--primary)" : "var(--muted-foreground)",
              opacity: played ? 0.95 : 0.35,
            }}
          />
        );
      })}
    </div>
  );
}
