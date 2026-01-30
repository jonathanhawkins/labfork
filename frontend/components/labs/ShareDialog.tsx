"use client";

/**
 * ShareDialog Component
 *
 * Dialog for sharing a lab with:
 * - Direct URL copy
 * - Embed code generation
 * - Social sharing links
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Share2,
  X,
  Link as LinkIcon,
  Code,
  Twitter,
  Copy,
  Check,
  ExternalLink,
  Mail,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";
import { getLabPath } from "@/lib/labs/types";

export interface ShareDialogProps {
  /** Lab to share */
  lab: Lab;
  /** Is dialog open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Custom class name */
  className?: string;
}

type ShareTab = "link" | "embed" | "social";

export function ShareDialog({
  lab,
  isOpen,
  onClose,
  className,
}: ShareDialogProps) {
  const [activeTab, setActiveTab] = useState<ShareTab>("link");
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [embedSize, setEmbedSize] = useState<"small" | "medium" | "large">("medium");

  // Get full URL
  const getFullUrl = useCallback(() => {
    if (typeof window === "undefined") return "";
    const basePath = getLabPath(lab.owner.username, lab.slug);
    return `${window.location.origin}${basePath}`;
  }, [lab.owner.username, lab.slug]);

  // Get embed code
  const getEmbedCode = useCallback(() => {
    const url = getFullUrl();
    const sizes = {
      small: { width: 400, height: 300 },
      medium: { width: 600, height: 450 },
      large: { width: 800, height: 600 },
    };
    const { width, height } = sizes[embedSize];
    return `<iframe src="${url}/embed" width="${width}" height="${height}" frameborder="0" allowfullscreen></iframe>`;
  }, [getFullUrl, embedSize]);

  // Copy to clipboard
  const copyToClipboard = useCallback(async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  }, []);

  // Social share URLs
  const getTwitterUrl = useCallback(() => {
    const url = getFullUrl();
    const text = `Check out "${lab.name}" on Voice Clone Lab!`;
    return `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(text)}`;
  }, [getFullUrl, lab.name]);

  const getEmailUrl = useCallback(() => {
    const url = getFullUrl();
    const subject = `Check out this lab: ${lab.name}`;
    const body = `I thought you might be interested in this lab:\n\n${lab.name}\n${lab.description || ""}\n\n${url}`;
    return `mailto:?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  }, [getFullUrl, lab.name, lab.description]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog */}
      <div
        className={cn(
          "relative w-full max-w-lg mx-4 bg-background-elevated rounded-lg border border-border shadow-xl",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Share2 className="w-5 h-5 text-foreground-bright" />
            <h2 className="text-lg font-medium text-foreground-bright">
              Share Lab
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-foreground-muted/10 transition-colors"
          >
            <X className="w-5 h-5 text-foreground-muted" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {[
            { id: "link" as const, label: "Link", icon: LinkIcon },
            { id: "embed" as const, label: "Embed", icon: Code },
            { id: "social" as const, label: "Social", icon: Share2 },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm transition-colors border-b-2 -mb-[1px]",
                activeTab === tab.id
                  ? "border-foreground-bright text-foreground-bright"
                  : "border-transparent text-foreground-muted hover:text-foreground"
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Link Tab */}
          {activeTab === "link" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-foreground-muted mb-2">
                  Lab URL
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    readOnly
                    value={getFullUrl()}
                    className="flex-1 px-3 py-2 rounded-lg text-sm bg-background border border-border text-foreground"
                  />
                  <button
                    onClick={() => copyToClipboard(getFullUrl(), "url")}
                    className={cn(
                      "px-3 py-2 rounded-lg border transition-colors flex items-center gap-2",
                      copiedField === "url"
                        ? "border-green-500/30 bg-green-500/10 text-green-400"
                        : "border-border hover:bg-foreground-muted/10"
                    )}
                  >
                    {copiedField === "url" ? (
                      <>
                        <Check className="w-4 h-4" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-foreground-muted/5 border border-border">
                <p className="text-xs text-foreground-muted">
                  Anyone with this link can view the lab
                  {lab.visibility === "private" && " (if they have access)"}
                  {lab.visibility === "unlisted" && " (unlisted - not searchable)"}.
                </p>
              </div>
            </div>
          )}

          {/* Embed Tab */}
          {activeTab === "embed" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-foreground-muted mb-2">
                  Embed Size
                </label>
                <div className="flex gap-2">
                  {(["small", "medium", "large"] as const).map((size) => (
                    <button
                      key={size}
                      onClick={() => setEmbedSize(size)}
                      className={cn(
                        "flex-1 px-3 py-2 rounded-lg text-sm border transition-colors capitalize",
                        embedSize === size
                          ? "border-foreground-bright bg-foreground-bright/10 text-foreground-bright"
                          : "border-border hover:bg-foreground-muted/10 text-foreground-muted"
                      )}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs text-foreground-muted mb-2">
                  Embed Code
                </label>
                <div className="relative">
                  <textarea
                    readOnly
                    value={getEmbedCode()}
                    rows={3}
                    className="w-full px-3 py-2 rounded-lg text-xs bg-background border border-border text-foreground font-mono"
                  />
                  <button
                    onClick={() => copyToClipboard(getEmbedCode(), "embed")}
                    className={cn(
                      "absolute top-2 right-2 px-2 py-1 rounded text-xs transition-colors flex items-center gap-1",
                      copiedField === "embed"
                        ? "bg-green-500/10 text-green-400"
                        : "bg-foreground-muted/10 hover:bg-foreground-muted/20 text-foreground-muted"
                    )}
                  >
                    {copiedField === "embed" ? (
                      <>
                        <Check className="w-3 h-3" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-foreground-muted/5 border border-border">
                <p className="text-xs text-foreground-muted">
                  Paste this code into your website to embed a live view of the lab.
                </p>
              </div>
            </div>
          )}

          {/* Social Tab */}
          {activeTab === "social" && (
            <div className="space-y-3">
              <a
                href={getTwitterUrl()}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 w-full p-3 rounded-lg border border-border hover:bg-foreground-muted/10 transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-[#1DA1F2]/10 flex items-center justify-center">
                  <Twitter className="w-5 h-5 text-[#1DA1F2]" />
                </div>
                <div className="flex-1 text-left">
                  <p className="text-sm font-medium text-foreground">Twitter</p>
                  <p className="text-xs text-foreground-muted">Share on Twitter</p>
                </div>
                <ExternalLink className="w-4 h-4 text-foreground-subtle" />
              </a>

              <a
                href={getEmailUrl()}
                className="flex items-center gap-3 w-full p-3 rounded-lg border border-border hover:bg-foreground-muted/10 transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
                  <Mail className="w-5 h-5 text-foreground-muted" />
                </div>
                <div className="flex-1 text-left">
                  <p className="text-sm font-medium text-foreground">Email</p>
                  <p className="text-xs text-foreground-muted">Share via email</p>
                </div>
                <ExternalLink className="w-4 h-4 text-foreground-subtle" />
              </a>

              <button
                onClick={() => copyToClipboard(getFullUrl(), "social")}
                className="flex items-center gap-3 w-full p-3 rounded-lg border border-border hover:bg-foreground-muted/10 transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
                  <LinkIcon className="w-5 h-5 text-foreground-muted" />
                </div>
                <div className="flex-1 text-left">
                  <p className="text-sm font-medium text-foreground">Copy Link</p>
                  <p className="text-xs text-foreground-muted">
                    {copiedField === "social" ? "Copied to clipboard!" : "Copy link to clipboard"}
                  </p>
                </div>
                {copiedField === "social" ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4 text-foreground-subtle" />
                )}
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 text-sm border border-border rounded-lg hover:bg-foreground-muted/10 transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

export default ShareDialog;
