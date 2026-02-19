import { useCallback, useRef, useState } from "react";
import { RotateCcw, AlertCircle } from "lucide-react";
import { Header } from "./components/Header";
import { ImageUploader } from "./components/ImageUploader";
import { BboxCanvas } from "./components/BboxCanvas";
import { ClassPill } from "./components/ClassPill";
import { ConceptPanel } from "./components/ConceptPanel";
import { DebugPanel } from "./components/DebugPanel";
import { Spinner } from "./components/Spinner";
import { PromptInput } from "./components/PromptInput";
import { PromptBar } from "./components/PromptBar";
import { ModelOutputWords } from "./components/ModelOutputWords";
import { useClassifyFlow } from "./hooks/useClassifyFlow";

function App() {
  const {
    state,
    upload,
    uploadSample,
    selectClass,
    selectWord,
    toggleConcept,
    reset,
  } = useClassifyFlow();

  // Keep a ref to the uploaded file so we can re-classify with a custom prompt
  const pendingFileRef = useRef<File | null>(null);

  /** Handle initial file upload — store file and classify with default prompt. */
  const handleUpload = useCallback(
    (file: File) => {
      pendingFileRef.current = file;
      upload(file);
    },
    [upload],
  );

  /** Handle custom-prompt classification. Re-uses the current file. */
  const handleClassifyWithPrompt = useCallback(
    (prompt: string) => {
      const file = pendingFileRef.current ?? state.imageFile;
      if (file) {
        upload(file, prompt);
      }
    },
    [upload, state.imageFile],
  );

  /** Handle word click from ModelOutputWords. */
  const handleWordClick = useCallback(
    (word: string) => {
      selectWord(word);
    },
    [selectWord],
  );

  const isProcessing =
    state.step === "classifying" ||
    state.step === "grounding" ||
    state.step === "explaining";

  return (
    <div className="max-w-6xl mx-auto px-4 py-4">
      <Header />

      {/* error banner */}
      {state.error && (
        <div className="flex items-center gap-2 bg-red-900/50 border border-red-700 rounded-md px-4 py-2 mb-4 text-sm text-red-200">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{state.error}</span>
          <button
            className="ml-auto text-red-400 hover:text-red-200 text-xs underline"
            onClick={reset}
          >
            Reset
          </button>
        </div>
      )}

      {/* upload state */}
      {state.step === "upload" && !state.imageUrl && (
        <ImageUploader
          onUpload={handleUpload}
          onSampleClick={uploadSample}
        />
      )}

      {/* main two-panel layout once we have an image */}
      {state.imageUrl && (
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,520px)_1fr] gap-6">
          {/* LEFT: image + bboxes */}
          <div className="space-y-3">
            {/* reset button */}
            <div className="flex items-center gap-2">
              <button
                className="flex items-center gap-1.5 rounded-md border border-slate-600 px-3 py-1 text-xs text-slate-400 hover:text-white hover:border-slate-400 transition-colors"
                onClick={reset}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                New image
              </button>

              {state.step === "classifying" && (
                <Spinner text="Classifying…" />
              )}
              {state.step === "grounding" && (
                <Spinner text="Detecting bounding boxes…" />
              )}
            </div>

            {/* image with bbox overlay */}
            <BboxCanvas
              imageUrl={state.imageUrl}
              objects={state.groundedObjects}
              nouns={state.nouns}
              selectedClass={state.selectedClass}
              onBoxClick={selectClass}
              maxWidth={500}
            />

            {/* Prompt bar — shows the prompts used */}
            <PromptBar
              classifyPrompt={state.classifyPrompt}
              groundingPrompt={state.groundingPrompt}
            />

            {/* Model output words — clickable tokens */}
            {state.modelOutput &&
              state.step !== "classifying" &&
              state.step !== "grounding" && (
                <ModelOutputWords
                  modelOutput={state.modelOutput}
                  nouns={state.nouns}
                  selectedWord={state.selectedClass}
                  onWordClick={handleWordClick}
                />
              )}

            {/* noun pills below model output */}
            {state.nouns.length > 0 &&
              state.step !== "classifying" &&
              state.step !== "grounding" && (
                <div className="space-y-1.5">
                  <p className="text-[0.65rem] uppercase tracking-wider text-slate-500 font-semibold">
                    {state.selectedClass
                      ? "Detected classes"
                      : "Click a bounding box or word to explain"}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {state.nouns.map((noun, i) => (
                      <ClassPill
                        key={noun}
                        label={noun}
                        color={
                          state.bboxColorsHex[
                            i % state.bboxColorsHex.length
                          ]
                        }
                        selected={state.selectedClass === noun}
                        onClick={() => selectClass(noun)}
                      />
                    ))}
                  </div>
                </div>
              )}

            {/* Custom prompt input — always visible once image is uploaded */}
            <div className="border-t border-slate-800 pt-3">
              <PromptInput
                onSubmit={handleClassifyWithPrompt}
                disabled={isProcessing}
              />
            </div>
          </div>

          {/* RIGHT: explanation panel */}
          <div className="min-w-0">
            {state.step === "explaining" && (
              <Spinner text={`Explaining "${state.selectedClass}"…`} />
            )}

            {state.explainResult && state.selectedClass && (
              <div className="space-y-2">
                <p className="text-[0.65rem] uppercase tracking-wider text-slate-500 font-semibold">
                  Concepts for &ldquo;{state.selectedClass}&rdquo;
                  <span className="normal-case tracking-normal font-normal ml-1">
                    — click a concept to show/hide it
                  </span>
                </p>
                <ConceptPanel
                  concepts={state.explainResult.top_concepts}
                  maskColorsHex={state.explainResult.mask_colors_hex}
                  activeConcepts={state.activeConcepts}
                  onToggleConcept={toggleConcept}
                />
              </div>
            )}

            {/* Debug panel */}
            {(state.modelOutput || state.explainResult) && (
              <DebugPanel
                modelOutput={state.modelOutput ?? undefined}
                explainResult={state.explainResult}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
