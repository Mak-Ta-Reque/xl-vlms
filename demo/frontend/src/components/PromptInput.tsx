import { useCallback, useState } from "react";
import { RotateCcw, Send } from "lucide-react";

const DEFAULT_PROMPT =
  "Name only the main objects in each part of the image. " +
  "Answer with a short comma-separated list of single words, no descriptions.";

interface Props {
  /** Called when user submits the prompt (triggers classify). */
  onSubmit: (prompt: string) => void;
  disabled?: boolean;
}

export function PromptInput({ onSubmit, disabled }: Props) {
  const [value, setValue] = useState(DEFAULT_PROMPT);

  const handleReset = useCallback(() => setValue(DEFAULT_PROMPT), []);
  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
  }, [value, onSubmit]);

  return (
    <div className="space-y-2">
      <p className="text-[0.65rem] uppercase tracking-wider text-slate-500 font-semibold">
        Classify prompt
      </p>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        rows={3}
        disabled={disabled}
        className="w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50 resize-y"
        placeholder="Enter a custom prompt…"
      />
      <div className="flex items-center gap-2">
        <button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Send className="h-3.5 w-3.5" />
          Classify with prompt
        </button>
        <button
          onClick={handleReset}
          disabled={disabled}
          className="flex items-center gap-1.5 rounded-md border border-slate-600 px-3 py-1.5 text-xs text-slate-400 hover:text-white hover:border-slate-400 disabled:opacity-40 transition-colors"
        >
          <RotateCcw className="h-3 w-3" />
          Reset default
        </button>
      </div>
    </div>
  );
}
