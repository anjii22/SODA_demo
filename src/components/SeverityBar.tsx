import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

interface SeverityBarProps {
  value: number;
  label: string;
  delay?: number;
}

const getSeverityInfo = (v: number) => {
  if (v <= 0.2) return { label: "Perfect", color: "#00d4aa", bg: "rgba(0,212,170,0.1)" };
  if (v <= 0.4) return { label: "Minor", color: "#4ade80", bg: "rgba(74,222,128,0.1)" };
  if (v <= 0.7) return { label: "Moderate", color: "#f59e0b", bg: "rgba(245,158,11,0.1)" };
  return { label: "Severe", color: "#ef4444", bg: "rgba(239,68,68,0.1)" };
};

export default function SeverityBar({ value, label, delay = 0 }: SeverityBarProps) {
  const info = getSeverityInfo(value);
  const percentage = Math.round(value * 100);
  const motionVal = useMotionValue(0);
  const springVal = useSpring(motionVal, { stiffness: 60, damping: 20 });
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (!hasAnimated.current) {
      hasAnimated.current = true;
      const timer = setTimeout(() => {
        motionVal.set(value * 100);
      }, delay * 1000);
      return () => clearTimeout(timer);
    }
  }, [value, delay, motionVal]);

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="space-y-2"
    >
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-400 font-medium">{label}</span>
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{ color: info.color, backgroundColor: info.bg }}
          >
            {info.label}
          </span>
          <span className="font-mono text-white text-xs">{percentage}%</span>
        </div>
      </div>

      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{
            width: springVal.get() + "%",
            backgroundColor: info.color,
            boxShadow: `0 0 8px ${info.color}66`,
          }}
          animate={{ width: `${percentage}%` }}
          transition={{ delay, duration: 1.2, ease: "easeOut" }}
        />
      </div>
    </motion.div>
  );
}
