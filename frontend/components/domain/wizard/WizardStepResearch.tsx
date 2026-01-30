"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { BookOpen, Search, Tag, X, Plus, Info } from "lucide-react";

/**
 * Research focus configuration
 */
export interface ResearchFocus {
  arxivCategories: string[];
  keywords: string[];
  tags: string[];
}

/**
 * Common arXiv categories for ML/AI research
 */
const ARXIV_CATEGORIES = [
  { id: "cs.LG", name: "Machine Learning", area: "CS" },
  { id: "cs.AI", name: "Artificial Intelligence", area: "CS" },
  { id: "cs.CL", name: "Computation and Language", area: "CS" },
  { id: "cs.CV", name: "Computer Vision", area: "CS" },
  { id: "cs.SD", name: "Sound", area: "CS" },
  { id: "cs.RO", name: "Robotics", area: "CS" },
  { id: "cs.NE", name: "Neural Computing", area: "CS" },
  { id: "stat.ML", name: "Machine Learning (Stats)", area: "Stats" },
  { id: "eess.AS", name: "Audio and Speech", area: "EESS" },
  { id: "q-fin.ST", name: "Statistical Finance", area: "Q-Fin" },
  { id: "q-bio.BM", name: "Biomolecules", area: "Q-Bio" },
  { id: "q-bio.NC", name: "Neurons and Cognition", area: "Q-Bio" },
];

/**
 * Suggested tags based on common research areas
 */
const SUGGESTED_TAGS = [
  "machine-learning",
  "deep-learning",
  "nlp",
  "speech",
  "voice",
  "tts",
  "asr",
  "prosody",
  "emotion",
  "trading",
  "finance",
  "robotics",
  "embodied-ai",
  "reinforcement-learning",
  "computer-vision",
  "biotech",
  "drug-discovery",
  "generative",
  "transformer",
  "diffusion",
];

export interface WizardStepResearchProps {
  /** Current research focus */
  research: ResearchFocus;
  /** Called when research focus changes */
  onResearchChange: (research: ResearchFocus) => void;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepResearch - Step 3: Set research focus
 */
export function WizardStepResearch({
  research,
  onResearchChange,
  className,
}: WizardStepResearchProps) {
  const [newKeyword, setNewKeyword] = useState("");
  const [newTag, setNewTag] = useState("");

  // Toggle arXiv category
  const toggleCategory = useCallback(
    (categoryId: string) => {
      const categories = research.arxivCategories.includes(categoryId)
        ? research.arxivCategories.filter((c) => c !== categoryId)
        : [...research.arxivCategories, categoryId];
      onResearchChange({ ...research, arxivCategories: categories });
    },
    [research, onResearchChange]
  );

  // Add keyword
  const addKeyword = useCallback(() => {
    const keyword = newKeyword.trim().toLowerCase();
    if (keyword && !research.keywords.includes(keyword)) {
      onResearchChange({
        ...research,
        keywords: [...research.keywords, keyword],
      });
      setNewKeyword("");
    }
  }, [newKeyword, research, onResearchChange]);

  // Remove keyword
  const removeKeyword = useCallback(
    (keyword: string) => {
      onResearchChange({
        ...research,
        keywords: research.keywords.filter((k) => k !== keyword),
      });
    },
    [research, onResearchChange]
  );

  // Toggle tag
  const toggleTag = useCallback(
    (tag: string) => {
      const tags = research.tags.includes(tag)
        ? research.tags.filter((t) => t !== tag)
        : [...research.tags, tag];
      onResearchChange({ ...research, tags });
    },
    [research, onResearchChange]
  );

  // Add custom tag
  const addCustomTag = useCallback(() => {
    const tag = newTag.trim().toLowerCase().replace(/\s+/g, "-");
    if (tag && !research.tags.includes(tag)) {
      onResearchChange({
        ...research,
        tags: [...research.tags, tag],
      });
      setNewTag("");
    }
  }, [newTag, research, onResearchChange]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-foreground-muted/10 mb-4">
          <BookOpen className="w-6 h-6 text-foreground-muted" />
        </div>
        <h2 className="text-lg font-normal text-foreground-bright">
          Research Focus
        </h2>
        <p className="text-sm text-foreground-muted mt-1">
          Configure arXiv categories and keywords for paper discovery
        </p>
      </div>

      {/* arXiv categories */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <BookOpen className="w-4 h-4" />
          arXiv Categories
          <span className="text-xs text-foreground-subtle">
            ({research.arxivCategories.length} selected)
          </span>
        </label>

        <div className="flex flex-wrap gap-2">
          {ARXIV_CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => toggleCategory(cat.id)}
              className={cn(
                "px-3 py-1.5 text-xs rounded-lg border transition-colors",
                research.arxivCategories.includes(cat.id)
                  ? "bg-foreground-bright/10 border-foreground-bright/50 text-foreground-bright"
                  : "bg-background-card border-border text-foreground-muted hover:border-foreground-muted"
              )}
            >
              <span className="font-mono">{cat.id}</span>
              <span className="ml-1.5 text-foreground-subtle">{cat.name}</span>
            </button>
          ))}
        </div>

        <p className="text-xs text-foreground-subtle flex items-center gap-1">
          <Info className="w-3 h-3" />
          Selected categories will be used to fetch relevant papers from arXiv
        </p>
      </div>

      {/* Keywords */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <Search className="w-4 h-4" />
          Search Keywords
        </label>

        {/* Add keyword input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newKeyword}
            onChange={(e) => setNewKeyword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addKeyword()}
            placeholder="Add a keyword..."
            className={cn(
              "flex-1 px-4 py-2 text-sm rounded-lg",
              "bg-background-card border border-border",
              "text-foreground placeholder:text-foreground-subtle",
              "focus:outline-none focus:border-foreground-muted"
            )}
          />
          <button
            onClick={addKeyword}
            disabled={!newKeyword.trim()}
            className={cn(
              "px-4 py-2 text-sm rounded-lg border transition-colors",
              "bg-background-card border-border text-foreground",
              "hover:bg-background-elevated disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Keywords list */}
        {research.keywords.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {research.keywords.map((keyword) => (
              <span
                key={keyword}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg bg-foreground-bright/10 border border-foreground-bright/30 text-foreground-bright"
              >
                {keyword}
                <button
                  onClick={() => removeKeyword(keyword)}
                  className="hover:text-white"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        )}

        <p className="text-xs text-foreground-subtle">
          Keywords refine paper searches within selected arXiv categories
        </p>
      </div>

      {/* Tags */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <Tag className="w-4 h-4" />
          Domain Tags
          <span className="text-xs text-foreground-subtle">
            ({research.tags.length} selected)
          </span>
        </label>

        {/* Suggested tags */}
        <div className="flex flex-wrap gap-2">
          {SUGGESTED_TAGS.map((tag) => (
            <button
              key={tag}
              onClick={() => toggleTag(tag)}
              className={cn(
                "px-2.5 py-1 text-xs rounded-lg border transition-colors",
                research.tags.includes(tag)
                  ? "bg-foreground-muted/20 border-foreground-muted/50 text-foreground"
                  : "bg-background-card border-border text-foreground-subtle hover:text-foreground-muted"
              )}
            >
              {tag}
            </button>
          ))}
        </div>

        {/* Custom tag input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addCustomTag()}
            placeholder="Add custom tag..."
            className={cn(
              "flex-1 px-3 py-1.5 text-xs rounded-lg",
              "bg-background-card border border-border",
              "text-foreground placeholder:text-foreground-subtle",
              "focus:outline-none focus:border-foreground-muted"
            )}
          />
          <button
            onClick={addCustomTag}
            disabled={!newTag.trim()}
            className={cn(
              "px-3 py-1.5 text-xs rounded-lg border transition-colors",
              "bg-background-card border-border text-foreground",
              "hover:bg-background-elevated disabled:opacity-50"
            )}
          >
            Add
          </button>
        </div>

        <p className="text-xs text-foreground-subtle">
          Tags help categorize your domain and make it discoverable
        </p>
      </div>
    </div>
  );
}

export { ARXIV_CATEGORIES, SUGGESTED_TAGS };
export default WizardStepResearch;
