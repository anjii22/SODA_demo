import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import type { TonguePositionEntry } from "../services/api";

interface TongueAnalysisProps {
  entries: TonguePositionEntry[];
}

const POSITION_COLORS: Record<string, string> = {
  bilabial: "#a78bfa",
  labiodental: "#c084fc",
  dental: "#60a5fa",
  alveolar: "#34d399",
  postalveolar: "#2dd4bf",
  palatal: "#facc15",
  velar: "#fb923c",
  glottal: "#f87171",
  uvular: "#f472b6",
  pharyngeal: "#e879f9",
};

const getPositionColor = (pos: string): string => {
  const lower = pos.toLowerCase();
  for (const [key, color] of Object.entries(POSITION_COLORS)) {
    if (lower.includes(key)) return color;
  }
  return "#94a3b8";
};

const PositionBadge = ({ position, dimmed = false }: { position: string; dimmed?: boolean }) => {
  const color = getPositionColor(position);
  return (
    <span
      className="text-xs font-semibold px-2.5 py-1 rounded-full capitalize"
      style={{
        color: dimmed ? "#64748b" : color,
        backgroundColor: dimmed ? "rgba(100,116,139,0.1)" : `${color}18`,
        border: `1px solid ${dimmed ? "#334155" : `${color}33`}`,
      }}
    >
      {position}
    </span>
  );
};

export default function TongueAnalysis({ entries }: TongueAnalysisProps) {
  if (!entries.length) return null;

  return (
    <div className="space-y-3">
      {entries.map((entry, i) => {
        const isSame =
          entry.expected_tongue_position.toLowerCase() ===
          entry.produced_tongue_position.toLowerCase();

        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.07, duration: 0.35 }}
            className={`glass-card rounded-xl p-4 border ${
              isSame ? "border-teal-500/15" : "border-amber-500/15"
            }`}
          >
            {/* Phoneme row */}
            <div className="flex items-center gap-2 text-xs text-slate-400 mb-3">
              <span className="font-mono font-bold text-teal-300 bg-teal-500/10 px-2 py-0.5 rounded">
                /{entry.expected_phoneme}/
              </span>
              <span>expected →</span>
              <span className="font-mono font-bold text-red-300 bg-red-500/10 px-2 py-0.5 rounded">
                /{entry.produced_phoneme}/
              </span>
              <span>produced</span>
            </div>

            {/* Position comparison */}
            <div className="flex items-center gap-3">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs text-slate-500">Target</span>
                <PositionBadge position={entry.expected_tongue_position} />
              </div>

              <ArrowRight
                className={`w-4 h-4 shrink-0 ${isSame ? "text-teal-500" : "text-slate-600"}`}
              />

              <div className="flex flex-col gap-0.5">
                <span className="text-xs text-slate-500">Actual</span>
                <PositionBadge
                  position={entry.produced_tongue_position}
                  dimmed={isSame}
                />
              </div>

              {isSame && (
                <span className="ml-auto text-xs text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded-full">
                  ✓ Correct
                </span>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
