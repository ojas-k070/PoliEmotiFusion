import { useEffect, useState } from "react";
import type { AnalysisResult, HistoryRecord, Modality } from "./types";

/* ------------------------------------------------------------------ */
/* Local history store                                                */
/* ------------------------------------------------------------------ */

const HISTORY_KEY = "pef.history.v2";
const SETTINGS_KEY = "pef.settings.v1";

let history: HistoryRecord[] = [];
let hydrated = false;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

function loadHistory() {
  if (hydrated || typeof window === "undefined") return;
  hydrated = true;
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as HistoryRecord[];
      if (Array.isArray(parsed)) history = parsed;
    }
  } catch {
    /* ignore corrupt storage */
  }
  emit();
}

function persist(records: HistoryRecord[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(records.slice(0, 50)));
  } catch {
    /* ignore quota errors */
  }
}

export function addAnalysis(result: AnalysisResult) {
  const record: HistoryRecord = {
    id: result.id,
    date: result.timestamp,
    modality: result.modality,
    inputLabel: result.inputLabel,
    emotion: result.emotion,
    confidence: result.confidence,
    category: result.category,
    status: "Completed",
    result,
  };
  history = [record, ...history];
  persist(history);
  emit();
}

export function clearOwnHistory() {
  history = [];
  if (typeof window !== "undefined") window.localStorage.removeItem(HISTORY_KEY);
  emit();
}

export function useHistory(): HistoryRecord[] {
  const [records, setRecords] = useState<HistoryRecord[]>(history);
  useEffect(() => {
    loadHistory();
    const listener = () => setRecords(history);
    listeners.add(listener);
    listener();
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return records;
}

/* ------------------------------------------------------------------ */
/* Settings                                                            */
/* ------------------------------------------------------------------ */

export type ThemeMode = "light" | "dark" | "system";

export interface Settings {
  theme: ThemeMode;
  animations: boolean;
  compact: boolean;
  notifications: boolean;
  defaultModality: Modality;
}

export const DEFAULT_SETTINGS: Settings = {
  theme: "system",
  animations: true,
  compact: false,
  notifications: true,
  defaultModality: "text",
};

let settings: Settings = DEFAULT_SETTINGS;
let settingsHydrated = false;
const settingsListeners = new Set<() => void>();

function applySettings(next: Settings) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const prefersDark =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  const dark = next.theme === "dark" || (next.theme === "system" && prefersDark);
  root.classList.toggle("dark", dark);
  root.classList.toggle("no-anim", !next.animations);
  root.classList.toggle("compact-ui", next.compact);
}

export function useSettings(): [Settings, (patch: Partial<Settings>) => void] {
  const [value, setValue] = useState<Settings>(settings);

  useEffect(() => {
    if (!settingsHydrated) {
      settingsHydrated = true;
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        if (raw) settings = { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
      } catch {
        /* ignore */
      }
      applySettings(settings);
    }
    const listener = () => setValue(settings);
    settingsListeners.add(listener);
    listener();
    return () => {
      settingsListeners.delete(listener);
    };
  }, []);

  const update = (patch: Partial<Settings>) => {
    settings = { ...settings, ...patch };
    try {
      window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
    applySettings(settings);
    settingsListeners.forEach((l) => l());
  };

  return [value, update];
}

export function notify(message: string) {
  return settings.notifications ? message : null;
}
