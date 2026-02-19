import { cn } from "../lib/utils";

interface Props {
  label: string;
  color: string;
  selected?: boolean;
  onClick?: () => void;
}

export function ClassPill({ label, color, selected, onClick }: Props) {
  return (
    <button
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-3 py-1 text-sm font-bold capitalize transition-all border-2",
        selected
          ? "scale-105 shadow-lg"
          : "opacity-70 hover:opacity-100",
      )}
      style={{
        color,
        borderColor: color,
        backgroundColor: selected ? `${color}15` : "transparent",
      }}
      onClick={onClick}
    >
      {label}
    </button>
  );
}
