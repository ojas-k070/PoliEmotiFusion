import { cn } from "@/lib/utils";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 40 40"
      role="img"
      aria-label="PoliEmotiFusion logo"
      className={cn("h-9 w-9", className)}
    >
      <defs>
        <linearGradient id="pef-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="oklch(0.62 0.15 250)" />
          <stop offset="100%" stopColor="oklch(0.42 0.14 262)" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="40" height="40" rx="11" fill="url(#pef-logo)" />
      <path
        d="M10 12.5h20a2.5 2.5 0 0 1 2.5 2.5v9a2.5 2.5 0 0 1-2.5 2.5H20l-6 4.5V26.5h-4A2.5 2.5 0 0 1 7.5 24v-9A2.5 2.5 0 0 1 10 12.5Z"
        fill="oklch(0.99 0.003 250)"
        opacity="0.95"
      />
      <circle cx="15.5" cy="18.5" r="1.6" fill="oklch(0.3 0.09 262)" />
      <circle cx="24.5" cy="18.5" r="1.6" fill="oklch(0.3 0.09 262)" />
      <path
        d="M14.5 22.4c1.6 1.9 3.4 2.8 5.5 2.8s3.9-.9 5.5-2.8"
        stroke="oklch(0.3 0.09 262)"
        strokeWidth="1.8"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

export function BrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <LogoMark />
      {!compact && (
        <div className="leading-tight">
          <div className="text-[15px] font-bold tracking-tight">PoliEmotiFusion</div>
          <div className="text-[11px] text-sidebar-foreground/60">
            Political Emotion Intelligence
          </div>
        </div>
      )}
    </div>
  );
}
