import { useCallback, useRef, useState } from "react";
import type {
  ClassifyResponse,
  ConceptInfo,
  ExplainResponse,
  FlowStep,
  GroundedObject,
} from "../types";
import * as api from "../lib/api";

export interface ClassifyFlowState {
  step: FlowStep;
  error: string | null;

  /* upload */
  imageFile: File | null;
  imageUrl: string | null; // local object URL for display

  /* classify */
  imageId: string | null;
  modelOutput: string | null;
  nouns: string[];

  /* prompts */
  classifyPrompt: string | null;
  groundingPrompt: string | null;
  /** true when the user supplied a custom prompt (skip bulk grounding) */
  isCustomPrompt: boolean;

  /* grounding */
  groundedObjects: GroundedObject[];

  /* explain */
  selectedClass: string | null;
  explainResult: ExplainResponse | null;
  bboxColorsHex: string[];

  /* concept toggle: set of concept_index values that are active */
  activeConcepts: Set<number>;
}

const INITIAL: ClassifyFlowState = {
  step: "upload",
  error: null,
  imageFile: null,
  imageUrl: null,
  imageId: null,
  modelOutput: null,
  nouns: [],
  classifyPrompt: null,
  groundingPrompt: null,
  isCustomPrompt: false,
  groundedObjects: [],
  selectedClass: null,
  explainResult: null,
  bboxColorsHex: ["#e67e22", "#27ae60", "#c0392b", "#2980b9", "#8e44ad"],
  activeConcepts: new Set(),
};

export function useClassifyFlow() {
  const [state, setState] = useState<ClassifyFlowState>(INITIAL);
  // cache binary results per label so re-clicking a bbox is instant
  const explainCache = useRef<Record<string, ExplainResponse>>({});

  const update = useCallback(
    (patch: Partial<ClassifyFlowState>) =>
      setState((prev) => ({ ...prev, ...patch })),
    [],
  );

  /** Upload an image → classify → ground. Full pipeline. Accepts optional custom prompt. */
  const upload = useCallback(
    async (file: File, customPrompt?: string) => {
      // reset
      explainCache.current = {};
      const localUrl = URL.createObjectURL(file);
      update({
        ...INITIAL,
        step: "classifying",
        imageFile: file,
        imageUrl: localUrl,
      });

      try {
        // 1. classify
        const cls: ClassifyResponse = await api.classifyImage(file, customPrompt);
        const usingCustom = !!customPrompt;
        update({
          imageId: cls.image_id,
          modelOutput: cls.model_output,
          nouns: cls.nouns,
          classifyPrompt: cls.prompt ?? null,
          isCustomPrompt: usingCustom,
          step: usingCustom ? "selecting" : "grounding",
        });

        // 2. ground all nouns ONLY when using the default prompt
        if (!usingCustom && cls.nouns.length > 0) {
          const gr = await api.groundImage(cls.image_id, cls.nouns);
          update({
            groundedObjects: gr.objects,
            groundingPrompt: gr.prompt ?? null,
            step: "selecting",
          });
        } else if (!usingCustom) {
          update({ step: "selecting" });
        }
      } catch (e: unknown) {
        update({
          error: e instanceof Error ? e.message : String(e),
          step: "upload",
        });
      }
    },
    [update],
  );

  /** User clicked a sample image → fetch as File then run upload(). */
  const uploadSample = useCallback(
    async (filename: string) => {
      try {
        const res = await fetch(`/samples/${filename}`);
        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type });
        await upload(file);
      } catch (e: unknown) {
        update({
          error: e instanceof Error ? e.message : String(e),
          step: "upload",
        });
      }
    },
    [upload, update],
  );

  /** User selected a class (bbox click or ClassPill). */
  const selectClass = useCallback(
    async (label: string) => {
      if (!state.imageId) return;

      // check cache
      const cached = explainCache.current[label];
      if (cached) {
        const allIndices = new Set(cached.top_concepts.map((c) => c.concept_index));
        update({
          selectedClass: label,
          explainResult: cached,
          activeConcepts: allIndices,
          step: "results",
        });
        return;
      }

      update({ selectedClass: label, step: "explaining", explainResult: null });

      try {
        const result = await api.explainClass(state.imageId, label);
        explainCache.current[label] = result;
        const allIndices = new Set(result.top_concepts.map((c) => c.concept_index));
        update({ explainResult: result, activeConcepts: allIndices, step: "results" });
      } catch (e: unknown) {
        update({
          error: e instanceof Error ? e.message : String(e),
          step: "selecting",
        });
      }
    },
    [state.imageId, update],
  );

  /**
   * User clicked a word in the model output.
   * Ground that single word (get bbox) → explain it (concept scoring).
   */
  const selectWord = useCallback(
    async (word: string) => {
      if (!state.imageId) return;

      // In custom-prompt mode, always ground the word (even if explain is cached)
      // so the bbox updates to the new word.
      const needsGround = state.isCustomPrompt;

      // Check explain cache
      const cached = explainCache.current[word];
      if (cached && !needsGround) {
        const allIndices = new Set(cached.top_concepts.map((c) => c.concept_index));
        update({
          selectedClass: word,
          explainResult: cached,
          activeConcepts: allIndices,
          step: "results",
        });
        return;
      }

      update({ selectedClass: word, step: "grounding", explainResult: null });

      try {
        // 1. Ground the single word to get a bbox
        const gr = await api.groundImage(state.imageId, [word]);
        if (gr.objects.length > 0) {
          setState((prev) => {
            if (prev.isCustomPrompt) {
              // Custom prompt: show ONLY this word's bbox
              return {
                ...prev,
                groundedObjects: gr.objects,
                nouns: [word],
                groundingPrompt: gr.prompt ?? prev.groundingPrompt,
              };
            }
            // Default prompt: keep all existing bboxes, merge new ones
            const existingNames = new Set(prev.groundedObjects.map((o) => o.name));
            const newObjs = gr.objects.filter((o) => !existingNames.has(o.name));
            const updatedNouns = prev.nouns.includes(word) ? prev.nouns : [...prev.nouns, word];
            return {
              ...prev,
              groundedObjects: [...prev.groundedObjects, ...newObjs],
              nouns: updatedNouns,
              groundingPrompt: gr.prompt ?? prev.groundingPrompt,
            };
          });
        }

        // 2. Explain (use cache if available)
        if (cached) {
          const allIndices = new Set(cached.top_concepts.map((c) => c.concept_index));
          update({ explainResult: cached, activeConcepts: allIndices, step: "results" });
        } else {
          update({ step: "explaining" });
          const result = await api.explainClass(state.imageId, word);
          explainCache.current[word] = result;
          const allIndices = new Set(result.top_concepts.map((c) => c.concept_index));
          update({ explainResult: result, activeConcepts: allIndices, step: "results" });
        }
      } catch (e: unknown) {
        update({
          error: e instanceof Error ? e.message : String(e),
          step: "selecting",
        });
      }
    },
    [state.imageId, update],
  );

  /** Toggle a concept active/inactive by its concept_index. */
  const toggleConcept = useCallback(
    (conceptIndex: number) => {
      setState((prev) => {
        const next = new Set(prev.activeConcepts);
        if (next.has(conceptIndex)) {
          next.delete(conceptIndex);
        } else {
          next.add(conceptIndex);
        }
        return { ...prev, activeConcepts: next };
      });
    },
    [],
  );

  /** Reset to initial state. */
  const reset = useCallback(() => {
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
    explainCache.current = {};
    setState(INITIAL);
  }, [state.imageUrl]);

  return { state, upload, uploadSample, selectClass, selectWord, toggleConcept, reset };
}
