import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, ImageIcon } from "lucide-react";
import { cn } from "../lib/utils";
import * as api from "../lib/api";

interface Props {
  onUpload: (file: File) => void;
  onSampleClick: (filename: string) => void;
  disabled?: boolean;
}

export function ImageUploader({ onUpload, onSampleClick, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [samples, setSamples] = useState<string[]>([]);

  useEffect(() => {
    api.getSamples().then((r) => setSamples(r.samples)).catch(() => {});
  }, []);

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (f && f.type.startsWith("image/")) onUpload(f);
    },
    [onUpload],
  );

  return (
    <div className="space-y-3">
      {/* drop zone */}
      <div
        className={cn(
          "relative flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 transition-colors cursor-pointer",
          drag
            ? "border-sky-400 bg-sky-400/10"
            : "border-slate-600 bg-slate-800/50 hover:border-slate-400",
          disabled && "opacity-50 pointer-events-none",
        )}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        <Upload className="h-8 w-8 text-slate-400" />
        <p className="text-sm text-slate-400">
          Drop an image here or <span className="text-sky-400 underline">browse</span>
        </p>
        <p className="text-xs text-slate-500">JPG, PNG, WEBP, BMP</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {/* sample images */}
      {samples.length > 0 && (
        <div>
          <p className="text-[0.65rem] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">
            Sample images
          </p>
          <div className="grid grid-cols-3 gap-2">
            {samples.map((fname) => (
              <button
                key={fname}
                className={cn(
                  "group relative overflow-hidden rounded-md border border-slate-700 hover:border-sky-400 transition-colors bg-slate-800",
                  disabled && "opacity-50 pointer-events-none",
                )}
                onClick={() => onSampleClick(fname)}
              >
                <img
                  src={`${api.BASE}/samples/${fname}`}
                  alt={fname}
                  className="w-full h-20 object-cover"
                  loading="lazy"
                />
                <span className="absolute inset-x-0 bottom-0 bg-black/60 text-[0.6rem] text-slate-300 px-1 py-0.5 truncate">
                  {fname.split(".")[0].slice(0, 20)}
                </span>
                <div className="absolute inset-0 bg-sky-400/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <ImageIcon className="h-5 w-5 text-sky-400" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
