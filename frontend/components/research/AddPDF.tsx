"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  FileText,
  Loader2,
  AlertCircle,
  Check,
  Upload,
  Edit3,
  User,
  BookOpen,
  Link2,
} from "lucide-react";
import type { Paper, PaperMetadata } from "@/lib/papers/types";
import type { PDFParseResult } from "@/lib/research/pdf-parser";

export interface AddPDFProps {
  /** Whether dialog is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Paper added handler */
  onPaperAdded?: (paper: Paper) => void;
  /** Custom class name */
  className?: string;
}

type DialogStep = "upload" | "parsing" | "review" | "result";

interface ManualFields {
  title: string;
  authors: string;
  abstract: string;
  doi: string;
  url: string;
}

/**
 * AddPDF - Dialog for uploading and parsing PDF papers
 */
export function AddPDF({
  isOpen,
  onClose,
  onPaperAdded,
  className,
}: AddPDFProps) {
  // State
  const [step, setStep] = useState<DialogStep>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parseResult, setParseResult] = useState<PDFParseResult | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [manualFields, setManualFields] = useState<ManualFields>({
    title: "",
    authors: "",
    abstract: "",
    doi: "",
    url: "",
  });
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset state when dialog closes
  useEffect(() => {
    if (!isOpen) {
      setStep("upload");
      setFile(null);
      setError(null);
      setParseResult(null);
      setPaper(null);
      setManualFields({
        title: "",
        authors: "",
        abstract: "",
        doi: "",
        url: "",
      });
    }
  }, [isOpen]);

  // Handle file selection
  const handleFileSelect = useCallback(
    async (selectedFile: File) => {
      // Validate file type
      if (!selectedFile.type.includes("pdf")) {
        setError("Please select a PDF file");
        return;
      }

      // Validate file size (50MB max)
      if (selectedFile.size > 50 * 1024 * 1024) {
        setError("File size must be less than 50MB");
        return;
      }

      setFile(selectedFile);
      setError(null);
      setStep("parsing");
      setIsLoading(true);

      try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch("/api/research/pdf", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (!data.success) {
          setError(data.error || "Failed to parse PDF");
          setStep("upload");
          return;
        }

        setParseResult(data.parseResult);

        // Pre-fill manual fields from parse result
        if (data.parseResult?.metadata) {
          const meta = data.parseResult.metadata;
          setManualFields({
            title: meta.title || "",
            authors: meta.authors?.map((a: { name: string }) => a.name).join(", ") || "",
            abstract: meta.abstract || "",
            doi: meta.doi || "",
            url: "",
          });
        }

        setStep("review");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to parse PDF");
        setStep("upload");
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  // Handle drag events
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        handleFileSelect(droppedFile);
      }
    },
    [handleFileSelect]
  );

  // Handle manual field changes
  const handleFieldChange = useCallback(
    (field: keyof ManualFields, value: string) => {
      setManualFields((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  // Submit paper
  const handleSubmit = useCallback(async () => {
    if (!manualFields.title.trim()) {
      setError("Title is required");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/research/pdf/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          parseResult,
          manualOverrides: {
            title: manualFields.title,
            authors: manualFields.authors.split(",").map((a) => a.trim()).filter(Boolean),
            abstract: manualFields.abstract,
            doi: manualFields.doi || undefined,
            url: manualFields.url || undefined,
          },
        }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Failed to add paper");
        return;
      }

      setPaper(data.paper);
      setStep("result");

      if (onPaperAdded) {
        onPaperAdded(data.paper);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add paper");
    } finally {
      setIsLoading(false);
    }
  }, [parseResult, manualFields, onPaperAdded]);

  // Handle keyboard
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    },
    [onClose]
  );

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={cn(
          "relative w-full max-w-lg mx-4 bg-background-card border border-border rounded-xl shadow-2xl",
          className
        )}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-foreground-muted" />
            </div>
            <div>
              <h2 className="text-lg font-normal text-foreground-bright">
                Upload PDF Paper
              </h2>
              <p className="text-xs text-foreground-muted">
                {step === "upload" && "Drag and drop or browse"}
                {step === "parsing" && "Extracting metadata..."}
                {step === "review" && "Review and edit details"}
                {step === "result" && "Paper added"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-foreground-muted hover:text-foreground p-2 rounded-lg hover:bg-foreground-muted/10"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Upload step */}
          {step === "upload" && (
            <div className="space-y-4">
              {/* Drop zone */}
              <div
                className={cn(
                  "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                  isDragging
                    ? "border-foreground-bright bg-foreground-bright/5"
                    : "border-border hover:border-foreground-muted"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={(e) =>
                    e.target.files?.[0] && handleFileSelect(e.target.files[0])
                  }
                  className="hidden"
                />
                <Upload className="w-10 h-10 mx-auto text-foreground-muted mb-4" />
                <p className="text-sm text-foreground">
                  {isDragging
                    ? "Drop PDF here"
                    : "Drag and drop a PDF or click to browse"}
                </p>
                <p className="text-xs text-foreground-muted mt-2">
                  Maximum file size: 50MB
                </p>
              </div>

              {/* Error message */}
              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
              )}

              {/* What we extract */}
              <div className="text-xs text-foreground-subtle">
                <p className="mb-2">We will extract:</p>
                <ul className="space-y-1 ml-4">
                  <li className="flex items-center gap-2">
                    <BookOpen className="w-3 h-3" />
                    Title and abstract
                  </li>
                  <li className="flex items-center gap-2">
                    <User className="w-3 h-3" />
                    Authors
                  </li>
                  <li className="flex items-center gap-2">
                    <Link2 className="w-3 h-3" />
                    DOI and arXiv ID (if present)
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* Parsing step */}
          {step === "parsing" && (
            <div className="flex flex-col items-center py-8 space-y-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full bg-foreground-bright/10 flex items-center justify-center">
                  <FileText className="w-8 h-8 text-foreground-bright animate-pulse" />
                </div>
                <div className="absolute inset-0 rounded-full border-2 border-foreground-bright/30 border-t-foreground-bright animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-foreground-bright">Parsing PDF...</p>
                <p className="text-sm text-foreground-muted mt-1">
                  {file?.name}
                </p>
              </div>
            </div>
          )}

          {/* Review step */}
          {step === "review" && (
            <div className="space-y-4 max-h-[50vh] overflow-y-auto">
              {/* Parse status */}
              {parseResult && (
                <div
                  className={cn(
                    "flex items-center gap-2 p-3 rounded-lg text-sm",
                    parseResult.status === "success"
                      ? "bg-green-500/10 text-green-400"
                      : parseResult.status === "needs_manual"
                        ? "bg-yellow-500/10 text-yellow-400"
                        : "bg-red-500/10 text-red-400"
                  )}
                >
                  {parseResult.status === "success" && (
                    <>
                      <Check className="w-4 h-4" />
                      Metadata extracted successfully
                    </>
                  )}
                  {parseResult.status === "needs_manual" && (
                    <>
                      <Edit3 className="w-4 h-4" />
                      Some fields need manual entry
                    </>
                  )}
                  {parseResult.status === "error" && (
                    <>
                      <AlertCircle className="w-4 h-4" />
                      Could not extract metadata
                    </>
                  )}
                </div>
              )}

              {/* Form fields */}
              <div className="space-y-3">
                {/* Title */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    Title *
                  </label>
                  <input
                    type="text"
                    value={manualFields.title}
                    onChange={(e) => handleFieldChange("title", e.target.value)}
                    className={cn(
                      "w-full px-3 py-2 text-sm rounded-lg",
                      "bg-background border border-border",
                      "text-foreground placeholder:text-foreground-subtle",
                      "focus:outline-none focus:border-foreground-muted"
                    )}
                    placeholder="Paper title"
                  />
                </div>

                {/* Authors */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    Authors (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={manualFields.authors}
                    onChange={(e) =>
                      handleFieldChange("authors", e.target.value)
                    }
                    className={cn(
                      "w-full px-3 py-2 text-sm rounded-lg",
                      "bg-background border border-border",
                      "text-foreground placeholder:text-foreground-subtle",
                      "focus:outline-none focus:border-foreground-muted"
                    )}
                    placeholder="e.g., John Doe, Jane Smith"
                  />
                </div>

                {/* Abstract */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    Abstract
                  </label>
                  <textarea
                    value={manualFields.abstract}
                    onChange={(e) =>
                      handleFieldChange("abstract", e.target.value)
                    }
                    rows={3}
                    className={cn(
                      "w-full px-3 py-2 text-sm rounded-lg resize-none",
                      "bg-background border border-border",
                      "text-foreground placeholder:text-foreground-subtle",
                      "focus:outline-none focus:border-foreground-muted"
                    )}
                    placeholder="Paper abstract"
                  />
                </div>

                {/* DOI */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    DOI (optional)
                  </label>
                  <input
                    type="text"
                    value={manualFields.doi}
                    onChange={(e) => handleFieldChange("doi", e.target.value)}
                    className={cn(
                      "w-full px-3 py-2 text-sm rounded-lg",
                      "bg-background border border-border",
                      "text-foreground placeholder:text-foreground-subtle",
                      "focus:outline-none focus:border-foreground-muted"
                    )}
                    placeholder="e.g., 10.1234/example"
                  />
                </div>

                {/* URL */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    Paper URL (optional)
                  </label>
                  <input
                    type="text"
                    value={manualFields.url}
                    onChange={(e) => handleFieldChange("url", e.target.value)}
                    className={cn(
                      "w-full px-3 py-2 text-sm rounded-lg",
                      "bg-background border border-border",
                      "text-foreground placeholder:text-foreground-subtle",
                      "focus:outline-none focus:border-foreground-muted"
                    )}
                    placeholder="e.g., https://arxiv.org/abs/..."
                  />
                </div>
              </div>

              {/* Error message */}
              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
              )}
            </div>
          )}

          {/* Result step */}
          {step === "result" && paper && (
            <div className="space-y-4">
              {/* Success message */}
              <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <Check className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm text-green-400">Paper added</p>
                  <p className="text-xs text-foreground-muted mt-0.5">
                    {paper.metadata.title}
                  </p>
                </div>
              </div>

              {/* Paper details */}
              <div className="text-sm text-foreground bg-background/50 p-3 rounded-lg border border-border">
                <p className="font-medium">{paper.metadata.title}</p>
                {paper.metadata.authors.length > 0 && (
                  <p className="text-foreground-muted mt-1">
                    {paper.metadata.authors
                      .slice(0, 3)
                      .map((a) => a.name)
                      .join(", ")}
                    {paper.metadata.authors.length > 3 && " et al."}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-border bg-background/50">
          {step === "upload" && (
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
            >
              Cancel
            </button>
          )}

          {step === "parsing" && (
            <div className="w-full text-center text-sm text-foreground-muted">
              Extracting metadata...
            </div>
          )}

          {step === "review" && (
            <>
              <button
                onClick={() => {
                  setStep("upload");
                  setFile(null);
                  setParseResult(null);
                }}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Back
              </button>
              <button
                onClick={handleSubmit}
                disabled={!manualFields.title.trim() || isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Check className="w-4 h-4" />
                )}
                Add Paper
              </button>
            </>
          )}

          {step === "result" && (
            <>
              <button
                onClick={() => {
                  setStep("upload");
                  setFile(null);
                  setParseResult(null);
                  setPaper(null);
                  setManualFields({
                    title: "",
                    authors: "",
                    abstract: "",
                    doi: "",
                    url: "",
                  });
                }}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Add Another
              </button>
              <button
                onClick={onClose}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white"
                )}
              >
                Done
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default AddPDF;
