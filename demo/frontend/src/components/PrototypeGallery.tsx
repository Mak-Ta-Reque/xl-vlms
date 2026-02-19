import { useState } from "react";
import { cn } from "../lib/utils";
import type { PrototypeInfo } from "../types";

interface Props {
  prototypes: PrototypeInfo[];
}

/**
 * Renders prototype images. Click an image to toggle between
 * the segmentation-mask overlay and the clean version.
 */
export function PrototypeGallery({ prototypes }: Props) {
  // Track which prototypes are showing the mask (true) vs clean (false)
  const [showMask, setShowMask] = useState<boolean[]>(
    () => prototypes.map(() => true),
  );

  if (prototypes.length === 0) return null;

  const toggle = (idx: number) => {
    setShowMask((prev) => {
      const next = [...prev];
      next[idx] = !next[idx];
      return next;
    });
  };

  return (
    <div className="flex gap-1.5 overflow-x-auto py-1">
      {prototypes.map((p, i) => {
        const hasBoth = !!p.image_b64_clean;
        const masked = showMask[i] ?? true;
        const src = masked ? p.image_b64 : (p.image_b64_clean ?? p.image_b64);

        return (
          <div key={i} className="relative flex-shrink-0">
            <img
              src={src}
              alt={`Prototype ${i + 1}`}
              role={hasBoth ? "button" : undefined}
              tabIndex={hasBoth ? 0 : undefined}
              onClick={hasBoth ? () => toggle(i) : undefined}
              onKeyDown={hasBoth ? (e) => { if (e.key === "Enter" || e.key === " ") toggle(i); } : undefined}
              className={cn(
                "h-[100px] w-[140px] object-cover rounded border border-slate-700 transition-all",
                hasBoth && "cursor-pointer hover:ring-2 hover:ring-blue-400 hover:scale-105",
              )}
            />
            {/* tiny badge indicating mask state */}
            {hasBoth && (
              <span
                className={cn(
                  "absolute bottom-1 right-1 text-[0.55rem] font-semibold px-1 py-px rounded",
                  masked
                    ? "bg-blue-500/80 text-white"
                    : "bg-slate-700/80 text-slate-300",
                )}
              >
                {masked ? "mask" : "clean"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
