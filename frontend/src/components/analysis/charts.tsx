import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { EMOTION_COLORS } from "@/lib/emotions";
import type { Emotion, TimelinePoint } from "@/lib/types";

const axisStyle = { fontSize: 11, fill: "var(--muted-foreground)" };

const tooltipStyle = {
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    fontSize: 12,
    color: "var(--popover-foreground)",
  },
  labelStyle: { color: "var(--popover-foreground)", fontWeight: 600 },
};

export function ProbabilityChart({
  probabilities,
  height = 260,
}: {
  probabilities: Record<Emotion, number>;
  height?: number;
}) {
  const data = (Object.entries(probabilities) as [Emotion, number][])
    .map(([emotion, value]) => ({ emotion, value }))
    .sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--border)" />
        <XAxis type="number" domain={[0, 100]} unit="%" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="emotion" width={78} tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`, "Probability"]} />
        <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16}>
          {data.map((d) => (
            <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function EmotionDonut({
  data,
  height = 280,
}: {
  data: { emotion: Emotion; value: number }[];
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="emotion"
          innerRadius="55%"
          outerRadius="82%"
          paddingAngle={2}
          stroke="var(--card)"
        >
          {data.map((d) => (
            <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />
          ))}
        </Pie>
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Tooltip {...tooltipStyle} formatter={(v: number) => [`${v}%`, "Share"]} />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function TimelineChart({ timeline }: { timeline: TimelinePoint[] }) {
  const data = timeline.map((p) => ({ ...p }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="time" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis domain={[0, 100]} unit="%" tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip
          {...tooltipStyle}
          formatter={(v: number, _n, item) => [
            `${v}% · ${(item?.payload as TimelinePoint | undefined)?.emotion ?? ""}`,
            "Segment confidence",
          ]}
        />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="var(--chart-1)"
          strokeWidth={2.5}
          dot={{ r: 4, strokeWidth: 0, fill: "var(--chart-1)" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ModalityBarChart({ data }: { data: { modality: string; count: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="modality" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip {...tooltipStyle} />
        <Bar dataKey="count" fill="var(--chart-1)" radius={[6, 6, 0, 0]} barSize={44} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function EmotionTrendChart({
  data,
}: {
  data: { week: string; Anger: number; Joy: number; Sadness: number; Neutral: number }[];
}) {
  const keys: Emotion[] = ["Anger", "Joy", "Sadness", "Neutral"];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="week" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {keys.map((k) => (
          <Line
            key={k}
            type="monotone"
            dataKey={k}
            stroke={EMOTION_COLORS[k]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
