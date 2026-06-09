/**
 * MemoizedMarkdown — Efficient streaming markdown renderer.
 *
 * Splits content into paragraph blocks and memoizes completed blocks
 * so only the in-progress (last) block re-renders on each token.
 *
 * Handles incomplete LaTeX ($$), code fences (```), and tables
 * by sanitizing in-progress content to prevent render errors.
 */
import { memo, useMemo } from "react";
import type { ChatSourceChunk, ChatImageRef } from "@/types";

// ---------------------------------------------------------------------------
// Block splitting — separate content into completed + in-progress blocks
// ---------------------------------------------------------------------------

interface SplitResult {
  completed: string[];
  inProgress: string;
}

/**
 * Split markdown content into paragraph blocks (by blank lines).
 * Tracks open code fences and LaTeX blocks to avoid splitting inside them.
 */
export function splitIntoBlocks(content: string): SplitResult {
  const lines = content.split("\n");
  const blocks: string[] = [];
  let currentBlock: string[] = [];
  let inCodeFence = false;
  let inLatexBlock = false;

  for (const line of lines) {
    // Track code fence state
    if (line.trimStart().startsWith("```")) {
      inCodeFence = !inCodeFence;
    }

    // Track $$ latex block state (only when not in code fence)
    if (!inCodeFence) {
      const trimmed = line.trim();
      // Match lines that start with $$ (opening/closing display math)
      if (trimmed === "$$" || (trimmed.startsWith("$$") && !trimmed.endsWith("$$"))) {
        inLatexBlock = !inLatexBlock;
      } else if (trimmed.endsWith("$$") && inLatexBlock) {
        inLatexBlock = false;
      }
    }

    // Blank line = paragraph boundary (only when not inside fence/latex)
    if (line.trim() === "" && !inCodeFence && !inLatexBlock) {
      if (currentBlock.length > 0) {
        blocks.push(currentBlock.join("\n"));
        currentBlock = [];
      }
      continue;
    }

    currentBlock.push(line);
  }

  // If we're inside an open fence/latex, the last block is in-progress
  if (inCodeFence || inLatexBlock || currentBlock.length > 0) {
    const inProgress = currentBlock.join("\n");
    return { completed: blocks, inProgress };
  }

  // All blocks completed (content ends with blank line or is empty)
  return { completed: blocks, inProgress: "" };
}

// ---------------------------------------------------------------------------
// Sanitize in-progress text — strip incomplete constructs
// ---------------------------------------------------------------------------

/**
 * Remove incomplete $$ blocks, ``` fences, and table rows
 * from in-progress text to prevent broken renders.
 */
export function sanitizeInProgress(text: string): string {
  if (!text) return "";

  let result = text;

  // Auto-close incomplete $$ block so KaTeX renders partial formula.
  const latexCount = (result.match(/\$\$/g) || []).length;
  if (latexCount % 2 !== 0) {
    const lastIdx = result.lastIndexOf("$$");
    const afterDollars = result.slice(lastIdx + 2);
    if (afterDollars.trim()) {
      result = result.slice(0, lastIdx) + "$$\n" + afterDollars.trimStart() + "\n$$";
    } else {
      result = result + "\n$$";
    }
  }

  // Auto-close incomplete ``` block so code renders with highlighting
  const fenceCount = (result.match(/```/g) || []).length;
  if (fenceCount % 2 !== 0) {
    result = result + "\n```";
  }

  // Strip incomplete table row (line starting with | but not ending with |)
  const lines = result.split("\n");
  while (lines.length > 0) {
    const last = lines[lines.length - 1];
    if (last.startsWith("|") && !last.trimEnd().endsWith("|")) {
      lines.pop();
    } else {
      break;
    }
  }
  result = lines.join("\n");

  return result.trimEnd();
}

// ---------------------------------------------------------------------------
// Simple fast hash for stable keys
// ---------------------------------------------------------------------------
function stableHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash.toString(36);
}

// ---------------------------------------------------------------------------
// Memoized single block — renders via the provided render function
// ---------------------------------------------------------------------------

interface MemoizedBlockProps {
  content: string;
  renderFn: (content: string) => React.ReactNode;
}

const MemoizedMarkdownBlock = memo(
  function MemoizedMarkdownBlock({ content, renderFn }: MemoizedBlockProps) {
    return <>{renderFn(content)}</>;
  },
  (prev, next) => prev.content === next.content && prev.renderFn === next.renderFn
);

// ---------------------------------------------------------------------------
// StreamingMarkdown — main export
// ---------------------------------------------------------------------------

export interface StreamingMarkdownProps {
  content: string;
  sources?: ChatSourceChunk[];
  imageRefs?: ChatImageRef[];
  isStreaming?: boolean;
  /** Render function for a single markdown block (uses MarkdownWithCitations) */
  renderBlock: (content: string) => React.ReactNode;
}

/**
 * Streaming-aware markdown renderer.
 *
 * Completed paragraph blocks are memoized (never re-render).
 * Only the in-progress block re-renders on each token.
 */
export function StreamingMarkdown({
  content,
  isStreaming = false,
  renderBlock,
}: StreamingMarkdownProps) {
  const { completed, inProgress } = useMemo(
    () => splitIntoBlocks(content),
    [content]
  );

  const sanitized = useMemo(
    () => (isStreaming ? sanitizeInProgress(inProgress) : inProgress),
    [inProgress, isStreaming]
  );

  return (
    <>
      {/* Completed blocks — fully memoized, never re-render */}
      {completed.map((block, i) => (
        <MemoizedMarkdownBlock
          key={`b-${stableHash(block)}-${i}`}
          content={block}
          renderFn={renderBlock}
        />
      ))}

      {/* In-progress block — re-renders on each token (parent mask handles fade) */}
      {sanitized ? renderBlock(sanitized) : null}
    </>
  );
}
