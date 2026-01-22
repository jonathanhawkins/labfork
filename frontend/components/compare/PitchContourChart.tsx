"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Activity } from "lucide-react";

interface PitchContourChartProps {
  contour: {
    times: number[];
    values: number[];
  };
  groundTruthPitch: number;
  title?: string;
}

export function PitchContourChart({
  contour,
  groundTruthPitch,
  title = "Pitch Contour",
}: PitchContourChartProps) {
  const data = contour.times.map((time, i) => ({
    time: time.toFixed(2),
    pitch: contour.values[i] > 0 ? contour.values[i] : null,
  }));

  const validPitches = contour.values.filter((v) => v > 0);
  const minPitch = Math.min(...validPitches, groundTruthPitch) * 0.8;
  const maxPitch = Math.max(...validPitches, groundTruthPitch) * 1.2;

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-white text-sm flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 10 }}
                label={{
                  value: "Time (s)",
                  position: "insideBottom",
                  offset: -5,
                  fill: "#9ca3af",
                  fontSize: 10,
                }}
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fill: "#9ca3af", fontSize: 10 }}
                domain={[minPitch, maxPitch]}
                label={{
                  value: "Hz",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#9ca3af",
                  fontSize: 10,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#fff",
                }}
                labelFormatter={(v) => `Time: ${v}s`}
                formatter={(v: number) => [`${v?.toFixed(1)} Hz`, "Pitch"]}
              />
              <ReferenceLine
                y={groundTruthPitch}
                stroke="#f97316"
                strokeDasharray="5 5"
                label={{
                  value: `GT: ${groundTruthPitch.toFixed(0)}Hz`,
                  fill: "#f97316",
                  fontSize: 10,
                  position: "right",
                }}
              />
              <Line
                type="monotone"
                dataKey="pitch"
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center justify-center gap-4 mt-2 text-xs text-zinc-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-purple-500 rounded"></span>
            Measured Pitch
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-orange-500 rounded border-dashed"></span>
            Ground Truth Mean
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
