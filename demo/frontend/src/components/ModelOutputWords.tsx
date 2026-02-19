import { cn } from "../lib/utils";

/** Vibrant word-token colours for extracted nouns. */
const WORD_COLORS = [
  { bg: "rgba(230, 126, 34, 0.18)", border: "#e67e22", text: "#f5c28a" },
  { bg: "rgba(39, 174, 96, 0.18)", border: "#27ae60", text: "#7ee2a8" },
  { bg: "rgba(192, 57, 43, 0.18)", border: "#c0392b", text: "#f1948a" },
  { bg: "rgba(41, 128, 185, 0.18)", border: "#2980b9", text: "#85c1e9" },
  { bg: "rgba(142, 68, 173, 0.18)", border: "#8e44ad", text: "#c39bd3" },
  { bg: "rgba(243, 156, 18, 0.18)", border: "#f39c12", text: "#f9d576" },
  { bg: "rgba(26, 188, 156, 0.18)", border: "#1abc9c", text: "#76d7c4" },
];

interface Props {
  /** Raw model output text. */
  modelOutput: string;
  /** Extracted nouns (used for colour-coding). */
  nouns?: string[];
  /** Currently selected / active word (if any). */
  selectedWord?: string | null;
  /** Called when user clicks a word. */
  onWordClick: (word: string) => void;
}

/**
 * Renders the model output as individually clickable word tokens.
 * Nouns are highlighted with vibrant pill colours matching the bbox palette.
 * All other words are subtly clickable on hover.
 */
export function ModelOutputWords({
  modelOutput,
  nouns = [],
  selectedWord,
  onWordClick,
}: Props) {
  if (!modelOutput) return null;

  // Build a set of normalised nouns for fast lookup
  const nounSet = new Set(nouns.map((n) => n.toLowerCase()));
  // Map each noun to a colour index
  const nounColorMap = new Map<string, number>();
  nouns.forEach((n, i) => nounColorMap.set(n.toLowerCase(), i % WORD_COLORS.length));

  // Split into tokens preserving whitespace/newlines for layout.
  const tokens = modelOutput.split(/(\s+)/);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <p className="text-[0.65rem] uppercase tracking-wider text-slate-500 font-semibold">
          Model output
        </p>
        <span className="text-[0.6rem] text-slate-500 font-normal">
          click any word to ground &amp; explain it
        </span>
      </div>
      <div className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 rounded-lg p-3 leading-[2] text-sm flex flex-wrap gap-y-0.5">
        {tokens.map((tok, i) => {
          // Pure whitespace → render as-is
          if (/^\s+$/.test(tok)) {
            return (
              <span key={i} className="whitespace-pre-wrap">
                {tok}
              </span>
            );
          }

          // Strip leading/trailing punctuation for the "clean" word value
          const clean = tok.replace(/^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$/g, "").toLowerCase();
          const isClickable = clean.length > 0;
          const isSelected = isClickable && selectedWord?.toLowerCase() === clean;
          const isNoun = isClickable && nounSet.has(clean);
          const colorIdx = isNoun ? nounColorMap.get(clean) ?? 0 : 0;
          const palette = WORD_COLORS[colorIdx];

          if (!isClickable) {
            return (
              <span key={i} className="text-slate-500">
                {tok}
              </span>
            );
          }

          // Selected word
          if (isSelected) {
            return (
              <span
                key={i}
                role="button"
                tabIndex={0}
                onClick={() => onWordClick(clean)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onWordClick(clean);
                }}
                className="cursor-pointer rounded-md px-1.5 py-0.5 font-semibold ring-2 ring-blue-400 transition-all"
                style={{
                  backgroundColor: "rgba(59, 130, 246, 0.3)",
                  color: "#93c5fd",
                }}
              >
                {tok}
              </span>
            );
          }

          // Noun word → coloured pill
          if (isNoun) {
            return (
              <span
                key={i}
                role="button"
                tabIndex={0}
                onClick={() => onWordClick(clean)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") onWordClick(clean);
                }}
                className="cursor-pointer rounded-md px-1.5 py-0.5 font-semibold border transition-all hover:scale-105 hover:shadow-lg"
                style={{
                  backgroundColor: palette.bg,
                  borderColor: palette.border,
                  color: palette.text,
                }}
              >
                {tok}
              </span>
            );
          }

          // Regular word → subtle but clickable
          return (
            <span
              key={i}
              role="button"
              tabIndex={0}
              onClick={() => onWordClick(clean)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onWordClick(clean);
              }}
              className={cn(
                "cursor-pointer rounded px-0.5 py-0.5 transition-all",
                "text-slate-300 hover:bg-slate-700/60 hover:text-white hover:rounded-md hover:px-1",
              )}
            >
              {tok}
            </span>
          );
        })}
      </div>
    </div>
  );
}
