import { ArrowLeft, Copy, Check } from "lucide-react";
import { useState } from "react";

/* ------------------------------------------------------------------ */
/*  Static endpoint definitions for the Classify API (port 8501)      */
/* ------------------------------------------------------------------ */

interface Endpoint {
  method: "GET" | "POST";
  path: string;
  description: string;
  requestBody?: string;
  responseBody: string;
  curlExample: string;
}

const ENDPOINTS: Endpoint[] = [
  {
    method: "GET",
    path: "/api/health",
    description: "Health-check endpoint. Returns a simple status object.",
    responseBody: `{
  "status": "ok"
}`,
    curlExample: `curl http://localhost:8501/api/health`,
  },
  {
    method: "GET",
    path: "/api/samples",
    description:
      "List all sample images available on the server (files in the demo/samples/ directory).",
    responseBody: `{
  "samples": ["cat.jpg", "dog.png", "car.webp"]
}`,
    curlExample: `curl http://localhost:8501/api/samples`,
  },
  {
    method: "POST",
    path: "/api/classify",
    description:
      "Upload an image (and an optional custom prompt) to run VLM classification. Returns the model's textual output and a list of extracted nouns.",
    requestBody: `Content-Type: multipart/form-data

Fields:
  file   : <binary image>     (required)
  prompt : "Describe this."   (optional, string)`,
    responseBody: `{
  "image_id": "abc123",
  "model_output": "A fluffy orange cat sitting on a couch.",
  "nouns": ["cat", "couch"],
  "prompt": "Describe what you see."
}`,
    curlExample: `curl -X POST http://localhost:8501/api/classify \\
  -F "file=@photo.jpg" \\
  -F "prompt=Describe what you see."`,
  },
  {
    method: "POST",
    path: "/api/ground",
    description:
      "Given an image_id (returned by /api/classify) and a list of nouns, run VLM grounding to obtain bounding boxes for each noun.",
    requestBody: `Content-Type: application/json

{
  "image_id": "abc123",
  "nouns": ["cat", "couch"]
}`,
    responseBody: `{
  "objects": [
    { "name": "cat",   "bbox": [52, 80, 310, 400] },
    { "name": "couch", "bbox": [0, 200, 500, 480] }
  ],
  "prompt": "Locate cat, couch in the image."
}`,
    curlExample: `curl -X POST http://localhost:8501/api/ground \\
  -H "Content-Type: application/json" \\
  -d '{"image_id":"abc123","nouns":["cat","couch"]}'`,
  },
  {
    method: "POST",
    path: "/api/explain",
    description:
      "Run binary concept scoring for a given label on a previously uploaded image (requires an image_id from /api/classify). Returns the top concepts with similarity scores and prototype images (base64).",
    requestBody: `Content-Type: application/json

{
  "image_id": "abc123",
  "label": "cat"
}`,
    responseBody: `{
  "model_output": "...",
  "top_concepts": [
    {
      "rank": 1,
      "similarity": 0.87,
      "concept_index": 42,
      "concept_names": ["furry", "whiskers"],
      "predictions": ["cat"],
      "prototypes": [
        { "image_b64": "<base64 string>" }
      ]
    }
  ],
  "mask_colors_hex": ["#ff0000", "#00ff00", "#0000ff"],
  "bbox_colors_hex": ["#ff0000", "#00ff00", "#0000ff"]
}`,
    curlExample: `curl -X POST http://localhost:8501/api/explain \\
  -H "Content-Type: application/json" \\
  -d '{"image_id":"abc123","label":"cat"}'`,
  },
  {
    method: "POST",
    path: "/api/explain-image",
    description:
      "Upload a fresh image and a label to run binary concept scoring directly — no prior /api/classify call needed. Returns the same response shape as /api/explain, plus the new image_id.",
    requestBody: `Content-Type: multipart/form-data

Fields:
  file  : <binary image>   (required)
  label : "cat"            (required, string)`,
    responseBody: `{
  "image_id": "def456.jpg",
  "model_output": "...",
  "top_concepts": [
    {
      "rank": 1,
      "similarity": 0.87,
      "concept_index": 42,
      "concept_names": ["furry", "whiskers"],
      "predictions": ["cat"],
      "prototypes": [
        { "image_b64": "<base64 string>" }
      ]
    }
  ],
  "mask_colors_hex": ["#ff0000", "#00ff00", "#0000ff"],
  "bbox_colors_hex": ["#ff0000", "#00ff00", "#0000ff"]
}`,
    curlExample: `curl -X POST http://localhost:8501/api/explain-image \\
  -F "file=@photo.jpg" \\
  -F "label=cat"`,
  },
  {
    method: "GET",
    path: "/api/image/{image_id}",
    description:
      "Retrieve a previously uploaded image as a base64-encoded JPEG data URI.",
    responseBody: `{
  "image_b64": "data:image/jpeg;base64,/9j/4AAQ..."
}`,
    curlExample: `curl http://localhost:8501/api/image/abc123`,
  },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function MethodBadge({ method }: { method: "GET" | "POST" }) {
  const color =
    method === "GET"
      ? "bg-emerald-600/80 text-emerald-100"
      : "bg-sky-600/80 text-sky-100";
  return (
    <span
      className={`${color} text-[0.65rem] font-bold tracking-wider px-2 py-0.5 rounded`}
    >
      {method}
    </span>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 p-1 rounded hover:bg-slate-600 text-slate-400 hover:text-white transition-colors"
      title="Copy"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-emerald-400" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

function CodeBlock({ label, code }: { label: string; code: string }) {
  return (
    <div className="mt-3">
      <p className="text-[0.6rem] uppercase tracking-widest text-slate-500 font-semibold mb-1">
        {label}
      </p>
      <div className="relative">
        <pre className="bg-slate-950 border border-slate-800 rounded-md p-3 text-xs text-slate-300 overflow-x-auto leading-relaxed whitespace-pre-wrap break-words">
          {code}
        </pre>
        <CopyButton text={code} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

interface ApiReferenceProps {
  onBack: () => void;
}

export function ApiReference({ onBack }: ApiReferenceProps) {
  return (
    <div className="space-y-6 pb-16">
      {/* sticky header */}
      <div className="sticky top-0 z-10 bg-[#0e1117]/90 backdrop-blur-sm py-3 -mx-4 px-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 rounded-md border border-slate-600 px-3 py-1.5 text-xs text-slate-400 hover:text-white hover:border-slate-400 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>
          <h2 className="text-white font-bold text-lg tracking-tight">
            API Reference
          </h2>
          <span className="text-slate-500 text-xs font-medium">
            Classify API &middot; port 8501
          </span>
        </div>
        <p className="text-slate-500 text-xs mt-1.5">
          Base URL: <code className="text-sky-400">http://localhost:8501</code>{" "}
          &mdash; {ENDPOINTS.length} endpoints
        </p>
      </div>

      {/* endpoint cards */}
      {ENDPOINTS.map((ep, idx) => (
        <div
          key={idx}
          className="bg-slate-900/60 border border-slate-800 rounded-lg p-5 space-y-2"
        >
          {/* title row */}
          <div className="flex items-center gap-2 flex-wrap">
            <MethodBadge method={ep.method} />
            <code className="text-sm text-white font-mono font-semibold">
              {ep.path}
            </code>
          </div>

          {/* description */}
          <p className="text-slate-400 text-sm leading-relaxed">
            {ep.description}
          </p>

          {/* request body */}
          {ep.requestBody && (
            <CodeBlock label="Request" code={ep.requestBody} />
          )}

          {/* response body */}
          <CodeBlock label="Response" code={ep.responseBody} />

          {/* curl example */}
          <CodeBlock label="Example (curl)" code={ep.curlExample} />
        </div>
      ))}
    </div>
  );
}
