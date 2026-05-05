import { useState, useCallback, useRef, DragEvent, ChangeEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Mic, FileAudio, X, Loader2, Zap } from "lucide-react";

interface UploadPanelProps {
  onAnalyze: (file: File, expectedText: string) => void;
  isLoading: boolean;
}

const ACCEPTED_TYPES = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"];
const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".ogg"];

export default function UploadPanel({ onAnalyze, isLoading }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [expectedText, setExpectedText] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (f: File): boolean => {
    const isValidType =
      ACCEPTED_TYPES.includes(f.type) ||
      ACCEPTED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext));
    if (!isValidType) {
      setFileError("Only .wav, .mp3, and .ogg files are supported.");
      return false;
    }
    if (f.size > 50 * 1024 * 1024) {
      setFileError("File size must be under 50 MB.");
      return false;
    }
    setFileError("");
    return true;
  };

  const handleFile = useCallback((f: File) => {
    if (validateFile(f)) setFile(f);
  }, []);

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile]
  );

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
    e.target.value = "";
  };

  const handleSubmit = () => {
    if (!file || !expectedText.trim()) return;
    onAnalyze(file, expectedText.trim());
  };

  const canSubmit = !!file && expectedText.trim().length > 0 && !isLoading;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-2xl mx-auto"
    >
      <div className="rounded-2xl glass p-6 space-y-5">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-xl bg-teal-500/10 border border-teal-500/20">
            <Mic className="w-5 h-5 text-teal-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Audio Analysis</h2>
            <p className="text-sm text-slate-400">Upload speech audio for phonological analysis</p>
          </div>
        </div>

        {/* Drop Zone */}
        <motion.div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => !file && fileInputRef.current?.click()}
          animate={{
            borderColor: isDragging
              ? "rgba(0,212,170,0.6)"
              : file
              ? "rgba(0,212,170,0.3)"
              : "rgba(255,255,255,0.08)",
            backgroundColor: isDragging
              ? "rgba(0,212,170,0.05)"
              : "rgba(15,23,42,0.5)",
          }}
          className="relative border-2 border-dashed rounded-xl p-8 cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-3 min-h-[160px]"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.ogg,audio/*"
            onChange={onFileChange}
            className="hidden"
          />

          <AnimatePresence mode="wait">
            {file ? (
              <motion.div
                key="file-info"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center gap-3 w-full"
              >
                <motion.div
                  animate={{ y: [0, -4, 0] }}
                  transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                >
                  <FileAudio className="w-10 h-10 text-teal-400" />
                </motion.div>
                <div className="text-center">
                  <p className="font-medium text-white text-sm truncate max-w-xs">{file.name}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{formatSize(file.size)}</p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setFileError("");
                  }}
                  className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors px-3 py-1 rounded-lg border border-red-500/20 hover:border-red-500/40"
                >
                  <X className="w-3 h-3" />
                  Remove file
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="upload-prompt"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-3 text-center"
              >
                <motion.div
                  animate={isDragging ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
                  transition={{ type: "spring", stiffness: 400, damping: 25 }}
                >
                  <Upload className="w-10 h-10 text-slate-500" />
                </motion.div>
                <div>
                  <p className="text-slate-300 font-medium">
                    {isDragging ? "Drop your audio file here" : "Drag & drop or click to upload"}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    Supports .wav, .mp3, .ogg — max 50 MB
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <AnimatePresence>
          {fileError && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="text-red-400 text-xs px-1"
            >
              {fileError}
            </motion.p>
          )}
        </AnimatePresence>

        {/* Expected Word Input */}
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400 inline-block" />
            Target Word
          </label>
          <input
            type="text"
            value={expectedText}
            onChange={(e) => setExpectedText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && canSubmit && handleSubmit()}
            placeholder="e.g. hello, cat, rabbit..."
            className="w-full bg-slate-900/70 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-teal-500/50 focus:ring-1 focus:ring-teal-500/30 transition-all text-sm"
          />
        </div>

        {/* Submit Button */}
        <motion.button
          onClick={handleSubmit}
          disabled={!canSubmit}
          whileHover={canSubmit ? { scale: 1.02 } : {}}
          whileTap={canSubmit ? { scale: 0.98 } : {}}
          className={`w-full py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2.5 transition-all duration-200 ${
            canSubmit
              ? "bg-teal-500 hover:bg-teal-400 text-slate-900 glow-teal"
              : "bg-slate-800 text-slate-500 cursor-not-allowed border border-white/5"
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing Speech...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Analyze Pronunciation
            </>
          )}
        </motion.button>
      </div>
    </motion.div>
  );
}
