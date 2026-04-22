import { useCallback, useEffect, useRef, useState } from "react";
import type { GroundedObject } from "../types";

/** Bbox palette (same as Python BBOX_COLORS_HEX). */
const BBOX_COLORS = [
  "#e67e22",
  "#27ae60",
  "#c0392b",
  "#2980b9",
  "#8e44ad",
];

interface Props {
  imageUrl: string;
  objects: GroundedObject[];
  bboxImageSize?: [number, number] | null;
  nouns: string[];
  selectedClass: string | null;
  onBoxClick: (noun: string) => void;
  maxWidth?: number;
}

export function BboxCanvas({
  imageUrl,
  objects,
  bboxImageSize,
  nouns,
  selectedClass,
  onBoxClick,
  maxWidth = 500,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{
    natW: number;
    natH: number;
    dispW: number;
    dispH: number;
  } | null>(null);
  const [hoveredNoun, setHoveredNoun] = useState<string | null>(null);

  // noun → color map
  const nounColor = useCallback(
    (noun: string) => {
      const idx = nouns.indexOf(noun);
      return BBOX_COLORS[(idx >= 0 ? idx : 0) % BBOX_COLORS.length];
    },
    [nouns],
  );

  // On image load, compute display size (fit to maxWidth) and set canvas dims
  const handleImgLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const scale = Math.min(1, maxWidth / natW);
    const dispW = Math.round(natW * scale);
    const dispH = Math.round(natH * scale);
    setImgSize({ natW, natH, dispW, dispH });
  }, [maxWidth]);

  // Resolve an object to its matched noun
  const matchNoun = useCallback(
    (obj: GroundedObject) => {
      for (const n of nouns) {
        if (n.includes(obj.name) || obj.name.includes(n)) return n;
      }
      return obj.name;
    },
    [nouns],
  );

  // Convert an object's bbox to clamped display-pixel coords
  const toDispCoords = useCallback(
    (obj: GroundedObject) => {
      if (!imgSize) return null;
      let [x1, y1, x2, y2] = obj.bbox;
      const sourceW = bboxImageSize?.[0] ?? imgSize.natW;
      const sourceH = bboxImageSize?.[1] ?? imgSize.natH;
      if ([x1, y1, x2, y2].every((v) => v >= 0 && v <= 1)) {
        x1 *= sourceW;
        y1 *= sourceH;
        x2 *= sourceW;
        y2 *= sourceH;
      }
      const scaleX = imgSize.dispW / sourceW;
      const scaleY = imgSize.dispH / sourceH;
      x1 *= scaleX; y1 *= scaleY; x2 *= scaleX; y2 *= scaleY;
      if (x1 > x2) [x1, x2] = [x2, x1];
      if (y1 > y2) [y1, y2] = [y2, y1];
      x1 = Math.max(0, x1);
      y1 = Math.max(0, y1);
      x2 = Math.min(imgSize.dispW, x2);
      y2 = Math.min(imgSize.dispH, y2);
      if (x2 - x1 < 2 || y2 - y1 < 2) return null;
      return { x1, y1, x2, y2 };
    },
    [imgSize, bboxImageSize],
  );

  // Draw bboxes on canvas overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgSize) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = imgSize.dispW;
    canvas.height = imgSize.dispH;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const obj of objects) {
      const d = toDispCoords(obj);
      if (!d) continue;
      const { x1, y1, x2, y2 } = d;

      const noun = matchNoun(obj);
      const color = nounColor(noun);
      const isSelected = selectedClass === noun;
      const isHovered = hoveredNoun === noun;

      // Semi-transparent fill on hover
      if (isHovered) {
        ctx.fillStyle = color + "20"; // ~12% opacity
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }

      // Draw rect — thin borders
      ctx.lineWidth = isSelected ? 2 : isHovered ? 1.5 : 1;
      ctx.strokeStyle = color;
      if (isHovered) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 3;
      }
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;

      // Label
      ctx.font = isHovered
        ? "bold 15px Inter, sans-serif"
        : "bold 14px Inter, sans-serif";
      const tw = ctx.measureText(noun).width;
      const labelH = 20;
      const labelY = Math.max(0, y1 - labelH - 2);

      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, tw + 8, labelH + 2);
      ctx.fillStyle = "#000";
      ctx.fillText(noun, x1 + 4, labelY + 15);
    }
  }, [objects, nouns, imgSize, nounColor, selectedClass, hoveredNoun, matchNoun, toDispCoords]);

  // Find which noun (if any) the cursor is over
  const hitTest = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): string | null => {
      if (!imgSize) return null;
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return null;
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      for (const obj of objects) {
        const d = toDispCoords(obj);
        if (!d) continue;
        if (cx >= d.x1 && cx <= d.x2 && cy >= d.y1 && cy <= d.y2) {
          return matchNoun(obj);
        }
      }
      return null;
    },
    [objects, imgSize, toDispCoords, matchNoun],
  );

  // Handle clicks on canvas
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const noun = hitTest(e);
      if (noun) onBoxClick(noun);
    },
    [hitTest, onBoxClick],
  );

  // Handle hover
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const noun = hitTest(e);
      setHoveredNoun(noun);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = noun ? "pointer" : "default";
      }
    },
    [hitTest],
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredNoun(null);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = "default";
    }
  }, []);

  return (
    <div ref={containerRef} className="relative inline-block">
      <img
        ref={(el) => {
          imgRef.current = el;
          if (el) el.onload = handleImgLoad;
        }}
        src={imageUrl}
        alt="Uploaded"
        style={{
          maxWidth: `${maxWidth}px`,
          display: "block",
        }}
        className="rounded-md"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0"
        style={{
          width: imgSize?.dispW ?? 0,
          height: imgSize?.dispH ?? 0,
        }}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}
