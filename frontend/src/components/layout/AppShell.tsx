import { Link, useRouterState } from "@tanstack/react-router";
import {
  BarChart3,
  FileText,
  History,
  Image as ImageIcon,
  Info,
  LayoutDashboard,
  Menu,
  Mic,
  Network,
  Settings as SettingsIcon,
  Video,
  X,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { BrandLockup } from "@/components/brand/Logo";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/text", label: "Text Analysis", icon: FileText },
  { to: "/image", label: "Image Analysis", icon: ImageIcon },
  { to: "/video", label: "Video Analysis", icon: Video },
  { to: "/audio", label: "Audio Analysis", icon: Mic },
  { to: "/history", label: "Analysis History", icon: History },
  { to: "/insights", label: "Insights", icon: BarChart3 },
  { to: "/architecture", label: "Architecture", icon: Network },
  { to: "/about", label: "About", icon: Info },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
] as const;

const MOBILE_NAV = NAV.filter((n) =>
  ["/", "/text", "/image", "/video", "/audio"].includes(n.to),
);

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-1 flex-col gap-1 overflow-y-auto px-3 py-4">
      {NAV.map(({ to, label, icon: Icon }) => (
        <Link
          key={to}
          to={to}
          onClick={onNavigate}
          activeOptions={{ exact: to === "/" }}
          className="group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-sidebar-foreground/75 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[status=active]:bg-sidebar-accent data-[status=active]:text-sidebar-accent-foreground data-[status=active]:shadow-sm"
        >
          <Icon className="h-[17px] w-[17px] shrink-0 opacity-80 group-data-[status=active]:text-sidebar-primary group-data-[status=active]:opacity-100" />
          <span className="truncate">{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <div className="min-h-screen bg-background">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[268px] flex-col bg-sidebar text-sidebar-foreground lg:flex">
        <div className="border-b border-sidebar-border px-5 py-5">
          <BrandLockup />
        </div>
        <NavList />
        <div className="border-t border-sidebar-border p-4 text-[11px] leading-relaxed text-sidebar-foreground/55">
          <p>PoliEmotiFusion · Political Emotion Intelligence</p>
        </div>
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            aria-label="Close navigation"
            className="absolute inset-0 bg-navy/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 flex w-[280px] flex-col bg-sidebar text-sidebar-foreground shadow-elevated">
            <div className="flex items-center justify-between border-b border-sidebar-border px-4 py-4">
              <BrandLockup />
              <Button
                variant="ghost"
                size="icon"
                aria-label="Close navigation"
                className="text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                onClick={() => setOpen(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            <NavList onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}

      <div className="lg:pl-[268px]">
        {/* Top bar */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-3 border-b border-border bg-background/85 px-4 backdrop-blur md:px-8">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              aria-label="Open navigation"
              className="lg:hidden"
              onClick={() => setOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <div className="lg:hidden">
              <BrandLockup compact />
            </div>
            <p className="hidden text-sm text-muted-foreground lg:block">
              Independent emotion analysis for political content
            </p>
          </div>
        </header>

        <main className="mx-auto w-full max-w-[1400px] px-4 pb-28 pt-6 md:px-8 md:pb-14 md:pt-8">
          {children}
        </main>
      </div>

      {/* Mobile bottom nav */}
      <nav className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-5 border-t border-border bg-background/95 backdrop-blur lg:hidden">
        {MOBILE_NAV.map(({ to, label, icon: Icon }) => (
          <Link
            key={to}
            to={to}
            activeOptions={{ exact: to === "/" }}
            className={cn(
              "flex flex-col items-center gap-1 py-2.5 text-[10px] font-medium text-muted-foreground transition-colors",
              "data-[status=active]:text-primary",
            )}
          >
            <Icon className="h-[18px] w-[18px]" />
            {label.replace(" Analysis", "")}
          </Link>
        ))}
      </nav>
    </div>
  );
}
