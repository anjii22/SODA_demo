import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RefreshCw,
  AlertTriangle,
  Wifi,
  ChevronDown,
  Zap,
  CheckCircle2,
  Upload,
  Waves,
  Type,
  ScanSearch,
  Crosshair,
  FileJson2,
  Mic,
  AudioLines,
  BarChart3,
  GitCompare,
  Cpu,
  Code2,
} from "lucide-react";
import { analyzeSpeech, ApiError, type AnalysisResult } from "./services/api";
import UploadPanel from "./components/UploadPanel";
import ResultsDashboard from "./components/ResultsDashboard";
import WaveformLoader from "./components/WaveformLoader";

type AppState = "idle" | "loading" | "result" | "error";

function PhonemePill({
  text,
  variant,
}: {
  text: string;
  variant: "expected" | "recognized" | "accent";
}) {
  const styles =
    variant === "expected"
      ? "bg-indigo-500/20 border-indigo-500 text-indigo-300"
      : variant === "recognized"
      ? "bg-amber-500/20 border-amber-500 text-amber-300"
      : "bg-teal-500/20 border-teal-500 text-teal-300";

  return (
    <span className={`inline-flex items-center justify-center font-mono-ui font-bold rounded-full border px-2 py-0.5 text-xs select-none whitespace-nowrap ${styles}`}>
      {text}
    </span>
  );
}

export default function App() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [lastExpectedText, setLastExpectedText] = useState("");

  const handleAnalyze = useCallback(async (file: File, expectedText: string) => {
    setAppState("loading");
    setResult(null);
    setErrorMessage("");
    setLastExpectedText(expectedText);

    try {
      const data = await analyzeSpeech(file, expectedText);
      setResult(data);
      setAppState("result");
    } catch (err) {
      if (err instanceof ApiError) {
        setErrorMessage(`API Error ${err.status}: ${err.message}`);
      } else if (err instanceof TypeError && (err.message.includes("fetch") || err.message.includes("network"))) {
        setErrorMessage(
          "Cannot reach the analysis server. Please check your network connection or try again later."
        );
      } else {
        setErrorMessage(err instanceof Error ? err.message : "An unexpected error occurred.");
      }
      setAppState("error");
    }
  }, []);

  const handleReset = () => {
    setAppState("idle");
    setResult(null);
    setErrorMessage("");
  };

  return (
    <div className="min-h-screen bg-navy-950 text-slate-100">
      {/* Hero */}
      <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-4">
        {/* Grid bg */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage:
              "linear-gradient(rgba(0, 212, 170, 0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 212, 170, 0.15) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
        {/* Glow blob */}
        <div
          className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none"
          style={{
            width: 600,
            height: 600,
            background: "radial-gradient(circle, rgba(0, 212, 170, 0.08) 0%, transparent 70%)",
          }}
        />

        <div className="relative z-10 max-w-4xl w-full text-center space-y-8">
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-teal-500/40 bg-teal-500/10 text-teal-400 text-sm font-mono-ui">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
              Research Preview
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.5 }}
            className="text-5xl md:text-7xl font-bold leading-tight tracking-tight"
          >
            <span className="text-white">SODA</span>
            <br />
            <span className="text-gradient-teal">Error Detection</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto leading-relaxed"
          >
            AI-powered pronunciation analysis with SODA-style error reporting for child speech therapy
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.5 }}
            className="flex justify-center"
          >
            <div
              className="relative flex items-center gap-4 px-8 py-4 rounded-2xl border border-navy-700"
              style={{ background: "rgba(17, 24, 39, 0.8)", backdropFilter: "blur(8px)" }}
            >
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
                <span className="text-xs font-mono-ui text-slate-500">LIVE INPUT</span>
              </div>
              <div className="flex items-end justify-center gap-[2px]" aria-hidden="true" style={{ height: 52 }}>
                {Array.from({ length: 22 }).map((_, i) => (
                  <div
                    key={i}
                    className="wave-bar rounded-t"
                    style={{
                      width: 2,
                      height: 6 + ((i * 13) % 64),
                      background: "rgb(0, 212, 170)",
                      opacity: 0.7 + (i % 3) * 0.1,
                      animationDuration: `${0.8 + (i % 5) * 0.15}s`,
                      transformOrigin: "center bottom",
                      animationDelay: `${(i % 14) * 0.07}s`,
                    }}
                  />
                ))}
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-mono-ui text-slate-500">16kHz</span>
                <div className="w-2 h-2 rounded-full bg-teal-500/40" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <button
              onClick={() => document.getElementById("pipeline")?.scrollIntoView({ behavior: "smooth" })}
              className="group inline-flex items-center gap-2 px-8 py-3.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-base transition-all duration-200 glow-teal"
            >
              See how it works
              <ChevronDown className="w-4 h-4 group-hover:translate-y-0.5 transition-transform" />
            </button>
            <a
              href="#demo"
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl border border-teal-500/40 text-teal-400 hover:bg-teal-500/10 font-semibold text-base transition-all duration-200"
            >
              Try demo
              <Zap className="w-4 h-4" />
            </a>
          </motion.div>

          <div className="flex items-center justify-center gap-1.5">
            <Wifi className="w-3 h-3 text-teal-500" />
            <span className="text-xs text-slate-500">
              API{" "}
              <span className="text-teal-500/80 font-mono-ui">/api/analyze</span>{" "}
              →{" "}
              <span className="text-teal-500/80 font-mono-ui">20.197.13.253:5050</span>
            </span>
          </div>
        </div>
      </section>

      {/* Pipeline explainer */}
      <section id="pipeline" className="py-24 px-4 max-w-5xl mx-auto">
        <div className="text-center mb-12 space-y-4">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-teal-500/30 bg-teal-500/10 text-teal-400 text-xs font-mono-ui">
            END-TO-END PIPELINE
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white">How it works</h2>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Nine stages transform a raw audio recording into a structured SODA error report.
          </p>
        </div>

        {/* Actions (optional) */}

        <div className="mb-8 h-1 bg-navy-700 rounded-full overflow-hidden">
          <div className="h-full bg-teal-500 rounded-full" style={{ boxShadow: "0 0 8px rgba(0, 212, 170, 0.5)" }} />
        </div>

        <div className="relative">
          {/* Vertical spine */}
          <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-teal-500/50 via-navy-700 to-transparent" />

          <div className="space-y-6">
            {[
              {
                title: "Audio Upload",
                subtitle: "Input received",
                detail: "Accept WAV/MP3/OGG and buffer for processing.",
                accent: "#00d4aa",
                icon: <Upload className="w-5 h-5" />,
              },
              {
                title: "Preprocessing",
                subtitle: "FFmpeg + denoise",
                detail: "Convert to 16kHz mono WAV + optional noise reduction.",
                accent: "#00d4aa",
                icon: <Waves className="w-5 h-5" />,
              },
              {
                title: "ASR (Word + Phoneme)",
                subtitle: "Wav2Vec2 + Groq Whisper (optional)",
                detail: "Decode phonemes and word hypotheses used for fusion + wrong-word checks.",
                accent: "#3b82f6",
                icon: <Mic className="w-5 h-5" />,
              },
              {
                title: "G2P Conversion",
                subtitle: "g2p_en",
                detail: "Expected text → ARPAbet phoneme tokens.",
                accent: "#ec4899",
                icon: <Type className="w-5 h-5" />,
              },
              {
                title: "Alignment",
                subtitle: "Needleman–Wunsch",
                detail: "Align expected vs produced phoneme sequences to locate mismatches.",
                accent: "#f59e0b",
                icon: <GitCompare className="w-5 h-5" />,
              },
              {
                title: "Error Detection",
                subtitle: "SODA + taxonomy",
                detail: "Classify Substitution / Omission / Distortion / Addition and generate error rows.",
                accent: "#fb7185",
                icon: <ScanSearch className="w-5 h-5" />,
              },
              {
                title: "Tongue Position",
                subtitle: "Articulation features",
                detail: "Summarize place/manner/voicing + tongue-position hints from feature diffs.",
                accent: "#34d399",
                icon: <Crosshair className="w-5 h-5" />,
              },
              {
                title: "Severity Scoring",
                subtitle: "0–1 severity scale",
                detail: "Compute overall + acoustic/phoneme/text severity values.",
                accent: "#a78bfa",
                icon: <BarChart3 className="w-5 h-5" />,
              },
              {
                title: "JSON Response",
                subtitle: "Client payload",
                detail: "Return structured results for UI visualization.",
                accent: "#60a5fa",
                icon: <FileJson2 className="w-5 h-5" />,
              },
            ].map((s, i) => (
              <motion.div
                key={s.title}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={{ delay: i * 0.04, duration: 0.35 }}
                className="relative pl-16"
              >
                {/* Stage marker */}
                <div className="absolute left-2 top-6">
                  <div
                    className="w-9 h-9 rounded-xl border flex items-center justify-center font-mono-ui text-xs font-bold"
                    style={{
                      background: "rgba(17, 24, 39, 0.95)",
                      borderColor: `${s.accent}55`,
                      color: s.accent,
                      boxShadow: `0 0 14px ${s.accent}20`,
                    }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </div>
                </div>

                <div
                  className="relative rounded-2xl border overflow-hidden transition-all duration-300 hover:border-navy-600"
                  style={{ background: "rgba(17, 24, 39, 0.95)", borderColor: "rgba(26, 37, 64, 1)" }}
                >
                  <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: s.accent }} />
                  <div className="p-5 md:p-6">
                    <div className="flex flex-col md:flex-row gap-6">
                      <div className="md:w-2/5 space-y-3">
                        <div className="flex items-center gap-3">
                          <div
                            className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0 border"
                            style={{ background: `${s.accent}18`, borderColor: `${s.accent}45`, color: s.accent }}
                          >
                            {s.icon}
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-mono-ui text-teal-500">STAGE {i + 1}</span>
                              <span
                                className="px-1.5 py-0.5 rounded text-xs font-mono-ui border"
                                style={{
                                  background: `${s.accent}1a`,
                                  borderColor: `${s.accent}33`,
                                  color: s.accent,
                                }}
                              >
                                ACTIVE
                              </span>
                            </div>
                            <h3 className="font-bold text-lg leading-tight text-white">{s.title}</h3>
                          </div>
                        </div>
                        <p className="text-sm leading-relaxed text-slate-300">{s.detail}</p>
                        <div className="text-xs font-mono-ui leading-relaxed p-3 rounded-lg border text-slate-400 bg-navy-800 border-navy-600">
                          {s.subtitle}
                        </div>
                      </div>

                      <div className="md:w-3/5 rounded-xl p-4 border bg-navy-900 border-navy-700">
                        <div className="space-y-3">
                          <div className="flex items-center gap-2 text-xs font-mono-ui text-slate-500">
                            <div className="w-2 h-2 rounded-full bg-teal-500/70 animate-pulse" />
                            <span>{s.title.toUpperCase()}</span>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <span
                              className="px-2.5 py-1 rounded-full text-xs font-mono-ui border"
                              style={{
                                background: `${s.accent}10`,
                                borderColor: `${s.accent}30`,
                                color: s.accent,
                              }}
                            >
                              {s.subtitle}
                            </span>
                            <span className="px-2.5 py-1 rounded-full text-xs font-mono-ui border border-navy-700 bg-navy-800 text-slate-400">
                              real API
                            </span>
                            <span className="px-2.5 py-1 rounded-full text-xs font-mono-ui border border-navy-700 bg-navy-800 text-slate-400">
                              latency-aware
                            </span>
                          </div>

                          <div className="h-1.5 bg-navy-700 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ background: s.accent }}
                              initial={{ width: "0%" }}
                              whileInView={{ width: "100%" }}
                              viewport={{ once: true }}
                              transition={{ duration: 0.9, delay: i * 0.03 }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="mt-12 p-6 rounded-2xl border border-teal-500/40 bg-teal-500/5 text-center">
          <div className="w-12 h-12 rounded-full bg-teal-500/20 border border-teal-500/40 flex items-center justify-center mx-auto mb-3">
            <CheckCircle2 className="w-6 h-6 text-teal-400" />
          </div>
          <h3 className="text-lg font-bold text-teal-300 mb-1">Pipeline ready</h3>
          <p className="text-sm text-slate-400">Scroll down to run the interactive live API demo.</p>
          <a
            href="#demo"
            className="inline-flex items-center gap-2 mt-4 px-5 py-2.5 rounded-xl bg-teal-500/20 border border-teal-500/40 text-teal-400 hover:bg-teal-500/30 text-sm font-semibold transition-all"
          >
            Try the demo
            <ChevronDown className="w-4 h-4" />
          </a>
        </div>
      </section>

      {/* Demo */}
      <section id="demo" className="py-24 px-4" style={{ background: "rgba(13, 20, 30, 0.6)" }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10 space-y-3">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 text-xs font-mono-ui">
              INTERACTIVE DEMO
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-white">Try it now</h2>
            <p className="text-slate-400 text-lg max-w-xl mx-auto">
              Upload your audio and call the live API. Results render from real backend output.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Left: uploader */}
            <div
              className="rounded-2xl border border-navy-700 p-6 space-y-5"
              style={{ background: "rgb(17, 24, 39)" }}
            >
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center">
                  <Zap className="w-4 h-4 text-teal-400" />
                </div>
                <h3 className="font-bold text-white">Live API call</h3>
              </div>

              <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-navy-600 bg-navy-900">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse shrink-0" />
                <span className="text-xs font-mono-ui text-slate-500 truncate">POST /api/analyze</span>
              </div>

              <div className="pt-2">
                <UploadPanel onAnalyze={handleAnalyze} isLoading={appState === "loading"} />
              </div>
            </div>

            {/* Right: status */}
            <div
              className="rounded-2xl border border-navy-700 p-6 space-y-5 flex flex-col min-h-[520px]"
              style={{ background: "rgb(17, 24, 39)" }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center">
                    <CheckCircle2 className="w-4 h-4 text-teal-400" />
                  </div>
                  <h3 className="font-bold text-white">Analysis Result</h3>
                </div>

                {result && (
                  <button
                    onClick={handleReset}
                    className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-navy-900 border border-navy-600 text-slate-400 hover:text-slate-200 text-xs font-mono-ui transition-colors"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Reset
                  </button>
                )}
              </div>

              <AnimatePresence mode="wait">
                {appState === "loading" && (
                  <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <WaveformLoader />
                  </motion.div>
                )}

                {appState === "error" && (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="rounded-2xl border border-red-500/20 bg-red-500/5 p-5"
                  >
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-xl bg-red-500/10">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                      </div>
                      <div className="min-w-0">
                        <p className="font-semibold text-white">Request failed</p>
                        <p className="text-sm text-slate-400 break-words">{errorMessage}</p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {appState === "idle" && (
                  <motion.div
                    key="idle"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex-1 flex flex-col items-center justify-center gap-3 py-12"
                  >
                    <div className="w-16 h-16 rounded-full border-2 border-dashed border-navy-600 flex items-center justify-center">
                      <ChevronDown className="w-7 h-7 text-slate-600 -rotate-90" />
                    </div>
                    <p className="text-sm text-slate-600 font-mono-ui">Awaiting analysis…</p>
                  </motion.div>
                )}

                {appState === "result" && result && (
                  <motion.div
                    key="result-ready"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="flex-1 flex flex-col items-center justify-center gap-3 py-12"
                  >
                    <div className="w-16 h-16 rounded-full border border-teal-500/30 bg-teal-500/10 flex items-center justify-center">
                      <CheckCircle2 className="w-7 h-7 text-teal-400" />
                    </div>
                    <p className="text-sm text-slate-300 font-mono-ui">Analysis ready — see full report below</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Full results below both cards */}
          <AnimatePresence>
            {appState === "result" && result && (
              <motion.div
                key="results-below"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
                transition={{ duration: 0.35 }}
                className="mt-8"
              >
                <ResultsDashboard result={result} expectedText={lastExpectedText} />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex items-center justify-center gap-2 mt-8 text-xs text-slate-500">
            <span className="font-mono-ui">Fields:</span>
            <span className="font-mono-ui text-teal-400">audio</span>
            <span>+</span>
            <span className="font-mono-ui text-teal-400">expected_text</span>
          </div>
        </div>
      </section>

      {/* Error types */}
      <section id="errors" className="py-24 px-4 max-w-6xl mx-auto">
        <div className="text-center mb-12 space-y-3">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-400 text-xs font-mono-ui">
            SODA ERROR TYPES
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-white">Error types</h2>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Four main speech-sound error types used in articulation-focused therapy reporting.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              color: "#6366f1",
              title: "Substitution",
              desc: "One target sound is replaced with another sound (e.g., /r/ produced as /w/).",
              expected: ["R"],
              recognized: ["W"],
            },
            {
              color: "#f59e0b",
              title: "Omission",
              desc: "A target sound is left out entirely (missing sound in the word).",
              expected: ["S", "K", "UW", "L"],
              recognized: ["K", "UW", "L"],
            },
            {
              color: "#00d4aa",
              title: "Distortion",
              desc:
                "The intended sound is produced in an imprecise way (recognizable, but “off” — e.g., a lisped /s/).",
              expected: ["S"],
              recognized: ["S*"],
              note:
                "Distortions are often described by articulatory detail (place/manner/voicing) rather than a different phoneme token.",
            },
            {
              color: "#ec4899",
              title: "Addition",
              desc: "An extra sound is inserted that wasn’t in the target word.",
              expected: ["K", "AE", "T"],
              recognized: ["K", "AE", "T", "S"],
            },
            ].map((card) => (
              <div
                key={card.title}
                className="rounded-2xl border border-navy-700 p-5 space-y-4 hover:border-navy-600 transition-all duration-300 group"
                style={{ background: "rgb(17, 24, 39)" }}
              >
                <div className="space-y-2">
                  <div className="w-8 h-1.5 rounded-full" style={{ background: card.color }} />
                  <h3 className="font-bold text-base text-white group-hover:text-slate-100 transition-colors">
                    {card.title}
                  </h3>
                </div>

                <p className="text-sm text-slate-400 leading-relaxed">{card.desc}</p>
              {"note" in card && card.note && (
                <p className="text-xs text-slate-500 leading-relaxed">{card.note}</p>
              )}

                <div className="space-y-2 pt-1">
                  <div className="text-xs font-mono-ui text-slate-600 uppercase tracking-wider">Example</div>

                  <div className="space-y-2">
                    <div className="space-y-1">
                      <div className="text-xs font-mono-ui text-slate-600">expected</div>
                      <div className="flex flex-wrap gap-1.5">
                        {card.expected.map((p) => (
                          <PhonemePill key={`e-${card.title}-${p}`} text={p} variant="expected" />
                        ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <div className="flex-1 h-px" style={{ background: `${card.color}40` }} />
                      <ChevronDown className="w-3 h-3 shrink-0" style={{ color: card.color }} />
                      <div className="flex-1 h-px" style={{ background: `${card.color}40` }} />
                    </div>

                    <div className="space-y-1">
                      <div className="text-xs font-mono-ui text-slate-600">recognized</div>
                      <div className="flex flex-wrap gap-1.5">
                        {card.recognized.map((p) => (
                          <PhonemePill key={`r-${card.title}-${p}`} text={p} variant="recognized" />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
      </section>

      {/* Technology */}
      <section id="stack" className="py-24 px-4" style={{ background: "rgba(13, 20, 30, 0.6)" }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12 space-y-3">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-slate-500/30 bg-slate-500/10 text-slate-400 text-xs font-mono-ui">
              TECHNOLOGY
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-white">Built with</h2>
            <p className="text-slate-400 text-lg max-w-xl mx-auto">
              State-of-the-art models and algorithms, combined into a production-ready pipeline.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {[
                {
                brand: "Groq",
                title: "Whisper Large v3",
                subtitle: "Word ASR (optional fusion)",
                  desc:
                  "Optional third word hypothesis via Groq-hosted Whisper (used for wrong-word detection + phoneme fusion via G2P).",
                  color: "#6366f1",
                  icon: <Mic className="w-6 h-6" />,
                },
                {
                  brand: "Meta",
                title: "Wav2Vec2 (CTC)",
                subtitle: "Phoneme ASR + Word ASR",
                  desc:
                  "Phoneme-first decoding with a Wav2Vec2 phoneme CTC model, plus optional word CTC for a second transcript stream.",
                  color: "#3b82f6",
                  icon: <AudioLines className="w-6 h-6" />,
                },
                {
                  brand: "Python",
                  title: "Flask API",
                  subtitle: "API Layer",
                  desc:
                  "Flask serves `POST /analyze` (multipart upload) and returns a thin JSON response used by this React demo.",
                  color: "#00d4aa",
                  icon: <Code2 className="w-6 h-6" />,
                },
              {
                brand: "NLP",
                title: "g2p_en",
                subtitle: "Grapheme → phoneme",
                desc:
                  "Converts expected text (and ASR words) into ARPAbet-style phoneme tokens for alignment and SODA classification.",
                color: "#ec4899",
                icon: <Cpu className="w-6 h-6" />,
              },
                {
                  brand: "Algorithm",
                  title: "Needleman–Wunsch",
                  subtitle: "Sequence Alignment",
                  desc:
                    "Global alignment of expected vs recognized phonemes reveals insertions, deletions, and substitutions.",
                  color: "#f59e0b",
                  icon: <GitCompare className="w-6 h-6" />,
                },
                {
                brand: "Articulation",
                title: "Feature analysis",
                subtitle: "Place · Manner · Voicing",
                  desc:
                  "Articulation mismatches are summarized using articulatory features: where the constriction happens (place), how airflow is shaped (manner), and whether vocal folds vibrate (voicing).",
                  color: "#34d399",
                  icon: <BarChart3 className="w-6 h-6" />,
                },
              ].map((t) => (
                <div
                  key={t.title}
                  className="rounded-2xl border border-navy-700 p-6 space-y-4 hover:border-navy-600 transition-all duration-300 group"
                  style={{ background: "rgb(17, 24, 39)" }}
                >
                  <div className="flex items-start justify-between">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center border"
                      style={{ background: `${t.color}15`, borderColor: `${t.color}40`, color: t.color }}
                    >
                      {t.icon}
                    </div>
                    <span
                      className="px-2.5 py-1 rounded-full text-xs font-mono-ui border"
                      style={{ background: `${t.color}10`, borderColor: `${t.color}30`, color: t.color }}
                    >
                      {t.brand}
                    </span>
                  </div>

                  <div>
                    <h3 className="font-bold text-lg text-white font-mono-ui">{t.title}</h3>
                    <p className="text-xs font-mono-ui mt-0.5" style={{ color: t.color }}>
                      {t.subtitle}
                    </p>
                  </div>

                  <p className="text-sm text-slate-400 leading-relaxed">{t.desc}</p>
                </div>
              ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-navy-700 py-8 text-center text-slate-500 text-sm">
        <p>SODA Error Detection — AI Speech Therapy Pipeline Visualizer</p>
        <p className="mt-1 font-mono-ui text-xs text-teal-500">
          v1.0.0 · Wav2Vec2 · Groq Whisper · g2p_en · Needleman–Wunsch · Flask
        </p>
      </footer>
    </div>
  );
}
