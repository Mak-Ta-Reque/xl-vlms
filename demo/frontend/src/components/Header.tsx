import { Microscope, BookOpen } from "lucide-react";

interface HeaderProps {
  onApiDocsClick?: () => void;
  showingApiDocs?: boolean;
}

export function Header({ onApiDocsClick, showingApiDocs }: HeaderProps) {
  return (
    <header className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-lg px-5 py-3 mb-4 flex items-center gap-3 border border-sky-400/15">
      <Microscope className="h-5 w-5 text-sky-400" />
      <h1 className="text-white font-bold text-lg tracking-tight">
        CVT LVLM Explainer
      </h1>
      <span className="text-slate-500 text-xs font-medium ml-1">
        [Qwen2.5-VL-3B]
      </span>
      <span className="ml-auto text-slate-500 text-xs">
        Upload → Detect → Explain
      </span>
      {onApiDocsClick && (
        <button
          onClick={onApiDocsClick}
          className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            showingApiDocs
              ? "bg-sky-600 text-white border border-sky-500"
              : "border border-slate-600 text-slate-400 hover:text-white hover:border-slate-400"
          }`}
        >
          <BookOpen className="h-3.5 w-3.5" />
          API Docs
        </button>
      )}
    </header>
  );
}
