import { motion } from "framer-motion";
import {
  CheckCircle2,
  XCircle,
  AlertCircle,
  MessageSquareText,
  Activity,
  Layers,
  Shuffle,
  Minus,
  Plus,
  ArrowLeftRight,
} from "lucide-react";
import type { AnalysisResult } from "../services/api";
import SeverityBar from "./SeverityBar";
import ErrorCard from "./ErrorCard";
import TongueAnalysis from "./TongueAnalysis";
import JsonViewer from "./JsonViewer";
import PipelineView from "./PipelineView";

interface ResultsDashboardProps {
  result: AnalysisResult;
  expectedText: string;
}

const Section = ({
  title,
  icon,
  children,
  delay = 0,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  delay?: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.45 }}
    className="glass rounded-2xl p-5 space-y-4"
  >
    <div className="flex items-center gap-2.5">
      <div className="p-1.5 rounded-lg bg-teal-500/10 text-teal-400">{icon}</div>
      <h3 className="font-semibold text-white text-sm">{title}</h3>
    </div>
    {children}
  </motion.div>
);

const sodaIcon = (type: string) => {
  const t = type.toLowerCase();
  if (t.includes("substitut")) return <Shuffle className="w-4 h-4" />;
  if (t.includes("omit") || t.includes("delet")) return <Minus className="w-4 h-4" />;
  if (t.includes("add") || t.includes("insert")) return <Plus className="w-4 h-4" />;
  return <ArrowLeftRight className="w-4 h-4" />;
};

export default function ResultsDashboard({ result, expectedText }: ResultsDashboardProps) {
  const isCorrect = result.is_correct && !result.wrong_word;
  const sodaColor = result.soda_errors.primary_error_type.toLowerCase().includes("substitut")
    ? "#f59e0b"
    : result.soda_errors.primary_error_type.toLowerCase().includes("omit") ||
      result.soda_errors.primary_error_type.toLowerCase().includes("delet")
    ? "#ef4444"
    : "#a78bfa";

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="w-full space-y-4"
    >
      {/* Summary Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className={`glass rounded-2xl p-5 border ${
          isCorrect ? "border-teal-500/20" : "border-red-500/15"
        }`}
      >
        <div className="flex items-start gap-4">
          <div
            className={`p-3 rounded-xl shrink-0 ${
              isCorrect ? "bg-teal-500/10" : "bg-red-500/10"
            }`}
          >
            {isCorrect ? (
              <CheckCircle2 className="w-7 h-7 text-teal-400" />
            ) : (
              <XCircle className="w-7 h-7 text-red-400" />
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <h2
                className={`text-lg font-bold ${
                  isCorrect ? "text-teal-400" : "text-red-400"
                }`}
              >
                {isCorrect ? "Pronunciation Correct" : "Pronunciation Errors Detected"}
              </h2>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-3">
              {/* Expected */}
              <div className="bg-slate-900/60 rounded-xl px-3 py-2.5">
                <p className="text-xs text-slate-500 mb-0.5">Expected</p>
                <p className="font-mono font-bold text-teal-300 text-sm truncate">
                  "{expectedText}"
                </p>
              </div>

              {/* Wav2Vec2 prediction */}
              <div className="bg-slate-900/60 rounded-xl px-3 py-2.5">
                <p className="text-xs text-slate-500 mb-0.5">Wav2Vec2</p>
                <p
                  className={`font-mono font-bold text-sm truncate ${
                    result.predicted_word.toLowerCase() === expectedText.toLowerCase()
                      ? "text-teal-300"
                      : "text-red-300"
                  }`}
                >
                  "{result.predicted_word}"
                </p>
              </div>

              {/* Whisper prediction */}
              <div className="bg-slate-900/60 rounded-xl px-3 py-2.5">
                <p className="text-xs text-slate-500 mb-0.5">Whisper</p>
                <p
                  className={`font-mono font-bold text-sm truncate ${
                    result.whisper_word.toLowerCase() === expectedText.toLowerCase()
                      ? "text-teal-300"
                      : "text-red-300"
                  }`}
                >
                  "{result.whisper_word}"
                </p>
              </div>
            </div>

            {result.wrong_word && (
              <div className="mt-3 flex items-center gap-2 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                Wrong word detected — the spoken word does not match the expected target
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Severity */}
      <Section
        title="Severity Analysis"
        icon={<Activity className="w-4 h-4" />}
        delay={0.1}
      >
        <div className="space-y-3.5">
          <SeverityBar value={result.severity} label="Overall Severity" delay={0.15} />
          <SeverityBar value={result.severity_phoneme} label="Phoneme Severity" delay={0.25} />
          <SeverityBar value={result.severity_text} label="Text Severity" delay={0.3} />
        </div>

        {/* Severity legend */}
        <div className="flex flex-wrap gap-3 pt-1 text-xs">
          {[
            { range: "0–20%", label: "Perfect", color: "#00d4aa" },
            { range: "20–40%", label: "Minor", color: "#4ade80" },
            { range: "40–70%", label: "Moderate", color: "#f59e0b" },
            { range: "70–100%", label: "Severe", color: "#ef4444" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <span
                className="w-2 h-2 rounded-full inline-block"
                style={{ backgroundColor: item.color }}
              />
              <span style={{ color: item.color }}>{item.label}</span>
              <span className="text-slate-600">{item.range}</span>
            </div>
          ))}
        </div>
      </Section>

      {/* SODA Error Type */}
      <Section
        title="SODA Error Classification"
        icon={<MessageSquareText className="w-4 h-4" />}
        delay={0.15}
      >
        <div className="flex items-center gap-4">
          <div
            className="flex items-center gap-2.5 px-4 py-3 rounded-xl border"
            style={{
              backgroundColor: `${sodaColor}10`,
              borderColor: `${sodaColor}30`,
              color: sodaColor,
            }}
          >
            {sodaIcon(result.soda_errors.primary_error_type)}
            <div>
              <p className="font-bold text-sm">{result.soda_errors.primary_error_type}</p>
              <p className="text-xs opacity-70 capitalize">{result.soda_errors.base_soda_error}</p>
            </div>
          </div>

          <div className="text-xs text-slate-400 leading-relaxed">
            <p>
              <span className="text-white font-medium">Substitution</span> — wrong phoneme produced
            </p>
            <p>
              <span className="text-white font-medium">Omission</span> — phoneme skipped
            </p>
            <p>
              <span className="text-white font-medium">Addition</span> — extra phoneme inserted
            </p>
          </div>
        </div>
      </Section>

      {/* Articulation Errors */}
      {result.articulation_errors.length > 0 && (
        <Section
          title={`Articulation Errors (${result.articulation_errors.length})`}
          icon={<XCircle className="w-4 h-4" />}
          delay={0.2}
        >
          <div className="space-y-3">
            {result.articulation_errors.map((err, i) => (
              <ErrorCard key={i} error={err} index={i} />
            ))}
          </div>
        </Section>
      )}

      {/* Tongue Position */}
      {result.tongue_position_analysis.length > 0 && (
        <Section
          title="Tongue Position Analysis"
          icon={<Layers className="w-4 h-4" />}
          delay={0.25}
        >
          <TongueAnalysis entries={result.tongue_position_analysis} />
        </Section>
      )}

      {/* Pipeline */}
      <Section
        title="Processing Pipeline"
        icon={<Activity className="w-4 h-4" />}
        delay={0.3}
      >
        <PipelineView result={result} />
      </Section>

      {/* Raw JSON */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.4 }}
      >
        <JsonViewer data={result} />
      </motion.div>
    </motion.div>
  );
}
