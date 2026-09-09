import { useEffect, useState } from "react";

/** Renders a locale-formatted timestamp only after hydration to avoid SSR mismatches. */
export function LocalTime({ value, dateOnly = false }: { value: string; dateOnly?: boolean }) {
  const [text, setText] = useState("");

  useEffect(() => {
    const d = new Date(value);
    setText(dateOnly ? d.toLocaleDateString() : d.toLocaleString());
  }, [value, dateOnly]);

  return <span suppressHydrationWarning>{text || "—"}</span>;
}
