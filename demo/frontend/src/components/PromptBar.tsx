import { useState } from "react";
import { ChevronDown, ChevronRight, MessageSquareText } from "lucide-react";

interface Props {
  classifyPrompt?: string | null;
  groundingPrompt?: string | null;
}

export function PromptBar({ classifyPrompt, groundingPrompt }: Props) {
  const [open, setOpen] = useState(false);

  if (!classifyPrompt && !groundingPrompt) return null;

  return (
    <div className="border border-slate-700 rounded-md bg-slate-900/70 mt-2">
      <button
        className="flex items-center gap-2 px-3 py-1.5 w-full text-left text-xs text-slate-400 hover:text-slate-200 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <MessageSquareText className="h-3.5 w-3.5" />
        <span className="font-medium">Prompts used</span>
        {open ? (
          <ChevronDown className="h-3 w-3 ml-auto" />
        ) : (
          <ChevronRight className="h-3 w-3 ml-auto" />
        )}
      </button>
      {open && (
        <div className="px-3 pb-2.5 space-y-2 text-xs">
          {classifyPrompt && (
            <div>
              <p className="text-slate-500 uppercase tracking-wider text-[0.6rem] font-semibold mb-0.5">
                Classify prompt
              </p>
              <pre className="bg-slate-800 rounded p-2 text-slate-300 whitespace-pre-wrap break-words">
                {classifyPrompt}
              </pre>
            </div>
          )}
          {groundingPrompt && (
            <div>
              <p className="text-slate-500 uppercase tracking-wider text-[0.6rem] font-semibold mb-0.5">
                Grounding prompt
              </p>
              <pre className="bg-slate-800 rounded p-2 text-slate-300 whitespace-pre-wrap break-words">
                {groundingPrompt}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
