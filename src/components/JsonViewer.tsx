import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Copy, Check, Code2 } from "lucide-react";

interface JsonViewerProps {
  data: unknown;
}

const colorizeJson = (json: string): string => {
  return json
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(
      /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
      (match) => {
        let cls = "color:#60a5fa"; // number (blue)
        if (/^"/.test(match)) {
          if (/:$/.test(match)) {
            cls = "color:#94a3b8"; // key (slate)
          } else {
            cls = "color:#34d399"; // string (green)
          }
        } else if (/true|false/.test(match)) {
          cls = "color:#a78bfa"; // boolean (purple)
        } else if (/null/.test(match)) {
          cls = "color:#f87171"; // null (red)
        }
        return `<span style="${cls}">${match}</span>`;
      }
    );
};

export default function JsonViewer({ data }: JsonViewerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const jsonString = JSON.stringify(data, null, 2);
  const colored = colorizeJson(jsonString);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-card rounded-xl border border-white/5 overflow-hidden">
      <button
        onClick={() => setIsOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3.5 hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <Code2 className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">Raw JSON Response</span>
          <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">
            {jsonString.split("\n").length} lines
          </span>
        </div>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.25 }}
        >
          <ChevronDown className="w-4 h-4 text-slate-500" />
        </motion.div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            <div className="border-t border-white/5">
              <div className="flex justify-end px-4 py-2 bg-slate-900/50">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-teal-400 transition-colors"
                >
                  {copied ? (
                    <>
                      <Check className="w-3 h-3 text-teal-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy JSON
                    </>
                  )}
                </button>
              </div>
              <pre
                className="px-4 pb-4 text-xs leading-relaxed overflow-x-auto font-mono max-h-96 overflow-y-auto"
                dangerouslySetInnerHTML={{ __html: colored }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
