import { Microscope } from "lucide-react";

export function Header() {
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
    </header>
  );
}
