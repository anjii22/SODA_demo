import { motion } from "framer-motion";
import { AlertTriangle, ArrowRight } from "lucide-react";
import type { ArticulationError } from "../services/api";

interface ErrorCardProps {
  error: ArticulationError;
  index: number;
}

const DetailRow = ({
  label,
  from,
  to,
}: {
  label: string;
  from: string;
  to: string;
}) => (
  <div className="flex items-center gap-2 text-xs">
    <span className="text-slate-500 w-16 shrink-0">{label}</span>
    <span className="font-mono text-amber-300 bg-amber-500/10 px-1.5 py-0.5 rounded">
      {from}
    </span>
    <ArrowRight className="w-3 h-3 text-slate-500 shrink-0" />
    <span className="font-mono text-red-300 bg-red-500/10 px-1.5 py-0.5 rounded">{to}</span>
  </div>
);

const parseDetail = (detail?: string): { from: string; to: string } | null => {
  if (!detail) return null;
  const parts = detail.split("→").map((s) => s.trim());
  if (parts.length === 2) return { from: parts[0], to: parts[1] };
  const arrowParts = detail.split("->").map((s) => s.trim());
  if (arrowParts.length === 2) return { from: arrowParts[0], to: arrowParts[1] };
  return { from: detail, to: "" };
};

export default function ErrorCard({ error, index }: ErrorCardProps) {
  const mannerParsed = parseDetail(error.details.manner_error);
  const placeParsed = parseDetail(error.details.place_error);
  const voicingParsed = parseDetail(error.details.voicing_error);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08, duration: 0.4 }}
      className="glass-card rounded-xl p-4 border border-red-500/10 glow-red"
    >
      {/* Phoneme mismatch header */}
      <div className="flex items-start gap-3 mb-3">
        <div className="p-1.5 rounded-lg bg-red-500/10 shrink-0 mt-0.5">
          <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex flex-col items-center">
            <span className="text-xs text-slate-500 mb-0.5">Expected</span>
            <span className="font-mono text-lg font-bold text-teal-400 bg-teal-500/10 px-3 py-1 rounded-lg">
              /{error.expected}/
            </span>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-600 mt-4" />
          <div className="flex flex-col items-center">
            <span className="text-xs text-slate-500 mb-0.5">Produced</span>
            <span className="font-mono text-lg font-bold text-red-400 bg-red-500/10 px-3 py-1 rounded-lg">
              /{error.produced}/
            </span>
          </div>
        </div>
      </div>

      {/* Articulation details */}
      {(mannerParsed || placeParsed || voicingParsed) && (
        <div className="mt-3 pt-3 border-t border-white/5 space-y-2">
          {mannerParsed && (
            <DetailRow
              label="Manner"
              from={mannerParsed.from}
              to={mannerParsed.to}
            />
          )}
          {placeParsed && (
            <DetailRow
              label="Place"
              from={placeParsed.from}
              to={placeParsed.to}
            />
          )}
          {voicingParsed && (
            <DetailRow
              label="Voicing"
              from={voicingParsed.from}
              to={voicingParsed.to}
            />
          )}
        </div>
      )}
    </motion.div>
  );
}
