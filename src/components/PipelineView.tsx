import { motion } from "framer-motion";
import {
  Upload,
  Waves,
  Cpu,
  FileText,
  GitCompare,
  ScanSearch,
  BarChart3,
  CheckCircle2,
} from "lucide-react";
import type { AnalysisResult } from "../services/api";

interface PipelineViewProps {
  result: AnalysisResult;
}

interface PipelineStep {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  detail: string;
  status: "success" | "warning" | "error";
}

export default function PipelineView({ result }: PipelineViewProps) {
  const steps: PipelineStep[] = [
    {
      icon: <Upload className="w-4 h-4" />,
      title: "Audio Upload",
      subtitle: "Input received",
      detail: "Audio file accepted for processing",
      status: "success",
    },
    {
      icon: <Waves className="w-4 h-4" />,
      title: "Preprocessing",
      subtitle: "FFmpeg + noisereduce",
      detail: "Audio normalized, denoised and converted to 16kHz mono WAV",
      status: "success",
    },
    {
      icon: <Cpu className="w-4 h-4" />,
      title: "Phoneme ASR",
      subtitle: "Wav2Vec2 CTC",
      detail: `Produced phonemes detected: /${
        result.articulation_errors.map((e) => e.produced).join(" ") || "—"
      }/`,
      status: result.articulation_errors.length > 0 ? "warning" : "success",
    },
    {
      icon: <FileText className="w-4 h-4" />,
      title: "Word ASR",
      subtitle: "Wav2Vec2 + Whisper Large v3",
      detail: `Wav2Vec2 → "${result.predicted_word}" | Whisper → "${result.whisper_word}"`,
      status: result.wrong_word ? "error" : "success",
    },
    {
      icon: <GitCompare className="w-4 h-4" />,
      title: "G2P Conversion",
      subtitle: "g2p_en grapheme-to-phoneme",
      detail: `Expected phonemes extracted from target text`,
      status: "success",
    },
    {
      icon: <ScanSearch className="w-4 h-4" />,
      title: "Alignment",
      subtitle: "Needleman-Wunsch",
      detail: `${result.articulation_errors.length} phoneme mismatch(es) aligned and detected`,
      status: result.articulation_errors.length > 0 ? "warning" : "success",
    },
    {
      icon: <CheckCircle2 className="w-4 h-4" />,
      title: "Error Detection",
      subtitle: "SODA + Articulation",
      detail: `Primary: ${result.soda_errors.primary_error_type} (${result.soda_errors.base_soda_error})`,
      status: result.articulation_errors.length > 0 ? "error" : "success",
    },
    {
      icon: <BarChart3 className="w-4 h-4" />,
      title: "Severity Scoring",
      subtitle: "Acoustic + Phoneme + Text",
      detail: `Overall severity: ${Math.round(result.severity * 100)}% | Acoustic: ${Math.round(
        result.acoustic_severity * 100
      )}%`,
      status:
        result.severity > 0.7
          ? "error"
          : result.severity > 0.4
          ? "warning"
          : "success",
    },
  ];

  const statusColors = {
    success: { dot: "#00d4aa", line: "rgba(0,212,170,0.3)", bg: "rgba(0,212,170,0.08)" },
    warning: { dot: "#f59e0b", line: "rgba(245,158,11,0.3)", bg: "rgba(245,158,11,0.08)" },
    error: { dot: "#ef4444", line: "rgba(239,68,68,0.3)", bg: "rgba(239,68,68,0.08)" },
  };

  return (
    <div className="relative pl-6">
      {/* Vertical line */}
      <div className="absolute left-2.5 top-4 bottom-4 w-px bg-gradient-to-b from-teal-500/40 via-slate-700 to-slate-800" />

      <div className="space-y-3">
        {steps.map((step, i) => {
          const colors = statusColors[step.status];
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -15 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.07, duration: 0.35 }}
              className="relative flex gap-3"
            >
              {/* Dot */}
              <div
                className="absolute -left-6 top-3.5 w-3 h-3 rounded-full border-2 border-slate-900 z-10"
                style={{ backgroundColor: colors.dot }}
              />

              {/* Card */}
              <div
                className="flex-1 rounded-xl p-3.5 border"
                style={{
                  backgroundColor: colors.bg,
                  borderColor: colors.line,
                }}
              >
                <div className="flex items-start gap-2.5">
                  <div
                    className="p-1.5 rounded-lg mt-0.5 shrink-0"
                    style={{ backgroundColor: `${colors.dot}18`, color: colors.dot }}
                  >
                    {step.icon}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold text-white">{step.title}</span>
                      <span
                        className="text-xs px-1.5 py-0.5 rounded font-mono"
                        style={{ color: colors.dot, backgroundColor: `${colors.dot}15` }}
                      >
                        {step.subtitle}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{step.detail}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
