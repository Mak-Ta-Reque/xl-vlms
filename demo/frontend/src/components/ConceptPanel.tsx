import type { ConceptInfo } from "../types";
import { ConceptCard } from "./ConceptCard";

interface Props {
  concepts: ConceptInfo[];
  maskColorsHex?: string[];
  /** Set of concept_index values that are currently active/visible. */
  activeConcepts?: Set<number>;
  /** Called when user toggles a concept on/off. */
  onToggleConcept?: (conceptIndex: number) => void;
}

export function ConceptPanel({
  concepts,
  maskColorsHex,
  activeConcepts,
  onToggleConcept,
}: Props) {
  if (concepts.length === 0) {
    return (
      <p className="text-slate-500 text-sm italic">No concepts found.</p>
    );
  }

  return (
    <div className="space-y-1">
      {concepts.map((c, i) => (
        <ConceptCard
          key={c.concept_index}
          concept={c}
          colorHex={maskColorsHex?.[i]}
          active={activeConcepts ? activeConcepts.has(c.concept_index) : true}
          onToggle={
            onToggleConcept
              ? () => onToggleConcept(c.concept_index)
              : undefined
          }
        />
      ))}
    </div>
  );
}
