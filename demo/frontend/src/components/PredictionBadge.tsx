import { cn } from "../lib/utils";

interface Props {
  prediction: string;
}

export function PredictionBadge({ prediction }: Props) {
  const isYes = prediction.toLowerCase() === "yes";
  return (
    <span
      className={cn(
        "inline-block px-1.5 py-0.5 rounded text-[0.7rem] font-semibold",
        isYes
          ? "bg-green-900/80 text-green-200"
          : "bg-red-900/80 text-red-200",
      )}
    >
      {prediction}
    </span>
  );
}
