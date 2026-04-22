/** API response / data types matching the FastAPI classify_api.py schema. */

export interface ClassifyResponse {
  image_id: string;
  model_output: string;
  nouns: string[];
  /** The prompt that was actually sent to the VLM for classification. */
  prompt: string;
  downsample_ratio?: number;
  original_size?: [number, number];
  processed_size?: [number, number];
}

export interface GroundedObject {
  name: string;
  bbox: [number, number, number, number];
}

export interface GroundResponse {
  objects: GroundedObject[];
  /** The prompt used for grounding / bounding-box detection. */
  prompt?: string;
  bbox_space?: "processed" | "original";
  bbox_image_size?: [number, number] | null;
  original_size?: [number, number] | null;
  processed_size?: [number, number] | null;
  downsample_ratio?: number | null;
  sync_contract_version?: number;
}

export interface PrototypeInfo {
  image_b64: string;
  /** Clean version without segmentation mask overlay. */
  image_b64_clean?: string;
}

export interface ConceptInfo {
  rank: number;
  similarity: number;
  concept_index: number;
  concept_names: string[];
  predictions: string[];
  prototypes: PrototypeInfo[];
}

export interface ExplainResponse {
  model_output: string;
  top_concepts: ConceptInfo[];
  mask_colors_hex: string[];
  bbox_colors_hex: string[];
}

export interface SamplesResponse {
  samples: string[];
}

/** Flow steps the UI moves through */
export type FlowStep =
  | "upload"
  | "classifying"
  | "grounding"
  | "selecting"
  | "explaining"
  | "results";
