import type {
  ClassifyResponse,
  ExplainResponse,
  GroundResponse,
  SamplesResponse,
} from "../types";

const BASE = ""; // proxied by Vite dev server → http://localhost:8501

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** Upload an image and classify it. Optionally send a custom prompt. */
export async function classifyImage(
  file: File,
  customPrompt?: string,
): Promise<ClassifyResponse> {
  const form = new FormData();
  form.append("file", file);
  if (customPrompt) {
    form.append("prompt", customPrompt);
  }
  return request<ClassifyResponse>("/api/classify", {
    method: "POST",
    body: form,
  });
}

/** Ground detected nouns → bounding boxes. */
export async function groundImage(
  imageId: string,
  nouns: string[],
): Promise<GroundResponse> {
  return request<GroundResponse>("/api/ground", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, nouns }),
  });
}

/** Explain a selected class via binary concept scoring. */
export async function explainClass(
  imageId: string,
  label: string,
): Promise<ExplainResponse> {
  return request<ExplainResponse>("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, label }),
  });
}

/** List available sample image filenames. */
export async function getSamples(): Promise<SamplesResponse> {
  return request<SamplesResponse>("/api/samples");
}

/** Get a temp image as base64 data URI. */
export async function getImageB64(
  imageId: string,
): Promise<{ image_b64: string }> {
  return request<{ image_b64: string }>(`/api/image/${imageId}`);
}
