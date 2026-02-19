import { Eye, EyeOff } from "lucide-react";
import { cn } from "../lib/utils";
import type { ConceptInfo } from "../types";
import { PredictionBadge } from "./PredictionBadge";
import { PrototypeGallery } from "./PrototypeGallery";

/** Mask-overlay colours (same as vis_utils.py MASK_COLORS_HEX). */
const MASK_COLORS_HEX = ["#38bdf8", "#fb923c", "#a78bfa", "#34d399", "#fb7185"];

interface Props {
  concept: ConceptInfo;
  colorHex?: string;
  /** Whether this concept is toggled on (visible). */
  active?: boolean;
  /** Called when the user toggles the concept. */
  onToggle?: () => void;
}

export function ConceptCard({ concept, colorHex, active = true, onToggle }: Props) {
  const color = colorHex ?? MASK_COLORS_HEX[(concept.rank - 1) % MASK_COLORS_HEX.length];
  const pct = (concept.similarity * 100).toFixed(1);
  const names = [...new Set(concept.concept_names)].join(", ");

  return (
    <div
      className={cn(
        "space-y-1.5 mb-3 transition-all duration-200",
        !active && "opacity-35 grayscale",
      )}
    >
      {/* rank + similarity row — clickable to toggle */}
      <button
        type="button"
        className="flex items-center gap-2 bg-slate-800 rounded-md px-3 py-1.5 w-full text-left hover:bg-slate-700/80 transition-colors cursor-pointer"
        style={{ borderLeft: `3px solid ${color}` }}
        onClick={onToggle}
        title={active ? "Hide concept" : "Show concept"}
      >
        <span className="text-slate-400 text-sm font-bold min-w-[1.2rem]">
          Concept-{concept.rank}
        </span>
        <span
          className="font-mono text-sm font-semibold ml-auto"
          style={{ color }}
        >
          {pct}%
        </span>
        {active ? (
          <Eye className="h-3.5 w-3.5 text-slate-400" />
        ) : (
          <EyeOff className="h-3.5 w-3.5 text-slate-500" />
        )}
      </button>

      {/* concept name */}
      {names && (
        <p
          className="text-sm font-medium capitalize pl-2"
          style={{ color }}
        >
          {names}
        </p>
      )}

      {/* prediction pills */}
      {concept.predictions.length > 0 && (
        <div className="flex flex-wrap gap-0.5 pl-2">
          {concept.predictions.map((p, i) => (
            <PredictionBadge key={i} prediction={p} />
          ))}
        </div>
      )}

      {/* prototypes — click individual images to toggle mask on/off */}
      <div className="pl-2">
        <PrototypeGallery prototypes={concept.prototypes} />
      </div>
    </div>
  );
}
