import { useState } from "react";
import { Bug, ChevronDown, ChevronRight } from "lucide-react";
import type { ExplainResponse } from "../types";

interface Props {
  modelOutput?: string;
  explainResult?: ExplainResponse | null;
}

export function DebugPanel({ modelOutput, explainResult }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-slate-700 rounded-md bg-slate-900 mt-4">
      <button
        className="flex items-center gap-2 px-3 py-2 w-full text-left text-sm text-slate-400 hover:text-slate-200 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <Bug className="h-4 w-4" />
        <span className="font-medium">Debug</span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 ml-auto" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 ml-auto" />
        )}
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-3 text-xs">
          {modelOutput && (
            <div>
              <p className="text-slate-500 uppercase tracking-wider text-[0.6rem] font-semibold mb-1">
                Model Output
              </p>
              <pre className="bg-slate-800 rounded p-2 text-slate-300 whitespace-pre-wrap break-words">
                {modelOutput}
              </pre>
            </div>
          )}
          {explainResult && (
            <div>
              <p className="text-slate-500 uppercase tracking-wider text-[0.6rem] font-semibold mb-1">
                Explain Result JSON
              </p>
              <pre className="bg-slate-800 rounded p-2 text-slate-300 whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
                {JSON.stringify(explainResult, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
