import { motion } from "framer-motion";

export default function WaveformLoader() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="flex flex-col items-center gap-6 py-12"
    >
      {/* Waveform bars */}
      <div className="flex items-center gap-1.5">
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className="wave-bar w-1.5 rounded-full"
            style={{
              height: "40px",
              backgroundColor: "#00d4aa",
              opacity: 0.7 + (i % 3) * 0.1,
              animationDelay: `${i * 0.1}s`,
            }}
          />
        ))}
      </div>

      <div className="text-center space-y-1">
        <p className="text-white font-semibold">Analyzing Pronunciation</p>
        <p className="text-sm text-slate-400">
          Running phoneme detection, alignment & scoring...
        </p>
      </div>

      {/* Progress steps */}
      <div className="flex flex-col gap-2 w-full max-w-xs">
        {[
          "Preprocessing audio...",
          "Running ASR models...",
          "Aligning phonemes...",
          "Computing severity...",
        ].map((step, i) => (
          <motion.div
            key={step}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.4, duration: 0.3 }}
            className="flex items-center gap-2 text-xs text-slate-400"
          >
            <motion.div
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.3 }}
              className="w-1.5 h-1.5 rounded-full bg-teal-400 shrink-0"
            />
            {step}
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
