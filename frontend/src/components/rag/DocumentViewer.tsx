import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { FileText, List, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { Document, ChatSourceChunk } from "@/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Heading {
  id: string;
  text: string;
  level: number;
}

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------
function ViewerSkeleton() {
  return (
    <div className="p-6 space-y-4">
      <div className="h-6 bg-muted rounded w-3/5" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-4/5" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-2/3" />
      <div className="h-20 bg-muted rounded w-full mt-4" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-3/4" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-1/2" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Error state
// ---------------------------------------------------------------------------
function ViewerError({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <FileText className="w-10 h-10 text-muted-foreground/40 mb-3" />
      <p className="text-sm font-medium">无法加载文档</p>
      <p className="text-xs text-muted-foreground mt-1 max-w-xs">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------
function ViewerEmpty() {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <FileText className="w-10 h-10 text-muted-foreground/30 mb-3" />
      <p className="text-sm text-muted-foreground">
        此文档暂无可用的解析内容
      </p>
      <p className="text-xs text-muted-foreground/60 mt-1">
        该文档可能尚未完成处理
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Table of Contents sidebar
// ---------------------------------------------------------------------------
const TOCSidebar = memo(function TOCSidebar({
  headings,
  activeId,
  onSelect,
}: {
  headings: Heading[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  if (headings.length === 0) return null;

  return (
    <nav className="w-52 flex-shrink-0 border-r overflow-y-auto py-3 px-2 hidden xl:block">
      <div className="flex items-center gap-1.5 px-2 mb-2">
        <List className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground tracking-wider">目录</span>
      </div>
      <ul className="space-y-0.5">
        {headings.map((h) => (
          <li key={h.id}>
            <button
              onClick={() => onSelect(h.id)}
              className={cn(
                "w-full text-left text-xs py-1 px-2 rounded-md transition-colors truncate",
                "hover:bg-muted",
                activeId === h.id
                  ? "text-primary font-medium bg-primary/10"
                  : "text-muted-foreground"
              )}
              style={{ paddingLeft: `${(h.level - 1) * 12 + 8}px` }}
              title={h.text}
            >
              {h.text}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
});

// ---------------------------------------------------------------------------
// Page divider — inserted between pages in the markdown
// ---------------------------------------------------------------------------
function PageDivider({ pageNo }: { pageNo: number }) {
  return (
    <div className="flex items-center gap-3 py-4 select-none" data-page={pageNo}>
      <div className="flex-1 border-t border-dashed border-muted-foreground/20" />
      <span className="text-[10px] font-medium text-muted-foreground/50 uppercase tracking-wider">
        第 {pageNo} 页
      </span>
      <div className="flex-1 border-t border-dashed border-muted-foreground/20" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Extract headings from markdown for TOC
// ---------------------------------------------------------------------------
function extractHeadings(markdown: string): Heading[] {
  const headings: Heading[] = [];
  const lines = markdown.split("\n");
  for (const line of lines) {
    const match = line.match(/^(#{1,4})\s+(.+)/);
    if (match) {
      const level = match[1].length;
      const text = match[2].replace(/[*_`#]/g, "").trim();
      const id = text
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .slice(0, 80);
      headings.push({ id, text, level });
    }
  }
  return headings;
}

// ---------------------------------------------------------------------------
// Insert page dividers into markdown text
// ---------------------------------------------------------------------------
const PAGE_TOKEN_PREFIX = "[EXPLORERAG_PAGE:";

/**
 * Docling exports a plain horizontal rule between PDF pages. Turn those
 * separators into deterministic tokens, which the paragraph renderer below
 * converts into real page anchors. This avoids page numbers drifting when
 * React renders the Markdown again.
 */
function insertPageDividers(markdown: string, pageCount?: number): string {
  let nextPage = 2;
  const maxPage = pageCount && pageCount > 0 ? pageCount : Number.POSITIVE_INFINITY;

  const withBreaks = markdown.replace(/^\s*---+\s*$/gm, (rule) => {
    if (nextPage > maxPage) return rule;
    const token = `${PAGE_TOKEN_PREFIX}${nextPage}]`;
    nextPage += 1;
    // The regex intentionally accepts Docling's surrounding whitespace, so
    // restore explicit paragraph boundaries for ReactMarkdown.
    return `\n\n${token}\n\n`;
  });

  // Page 1 has no preceding separator, but citations to it still need an anchor.
  return `${PAGE_TOKEN_PREFIX}1]\n\n${withBreaks}`;
}

function getPageToken(children: React.ReactNode): number | null {
  const match = getHeadingText(children)
    .trim()
    .match(/^\[EXPLORERAG_PAGE:(\d+)\]$/);
  return match ? Number(match[1]) : null;
}

// ---------------------------------------------------------------------------
// Citation source matching
// ---------------------------------------------------------------------------
function normalizeForMatch(value: string): string {
  return value
    // Preserve words when PDFs wrap a hyphenated English word over two lines.
    .replace(/-\s*\n\s*/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[`*_>#|]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLocaleLowerCase();
}

function getSourceFragments(content: string): string[] {
  const fragments = content
    .split(/\n+/)
    .map(normalizeForMatch)
    .filter((line) => line.length >= 24)
    .sort((a, b) => b.length - a.length);

  const whole = normalizeForMatch(content);
  if (whole.length >= 48) {
    // A source can span multiple rendered paragraphs. Keep a few long,
    // distinctive windows as a fallback for parsers that merge PDF lines.
    for (const start of [0, Math.floor(whole.length / 3), Math.floor((whole.length * 2) / 3)]) {
      const fragment = whole.slice(start, start + 180).trim();
      if (fragment.length >= 48) fragments.push(fragment);
    }
  }

  return [...new Set(fragments)].slice(0, 24);
}

function findChunkElement(root: HTMLElement, content: string, chunkId?: string): HTMLElement | null {
  if (chunkId) {
    const anchored = root.querySelector<HTMLElement>(
      `[data-chunk-id="${CSS.escape(chunkId)}"]`
    );
    if (anchored) return anchored;
  }
  const candidates = Array.from(
    root.querySelectorAll<HTMLElement>(
      "h1, h2, h3, h4, p, li, blockquote, pre, table, figure"
    )
  );
  const fragments = getSourceFragments(content);

  // Prefer an exact, distinctive source line. This handles tables,
  // paragraphs, figures, and source chunks without a heading path.
  for (const fragment of fragments) {
    const found = candidates.find((element) =>
      normalizeForMatch(element.textContent || "").includes(fragment)
    );
    if (found) {
      if (chunkId) found.dataset.chunkId = chunkId;
      return found;
    }
  }

  // Last-resort scoring keeps the interaction useful when PDF extraction has
  // minor typography differences between indexed and rendered text.
  const sourceTerms = normalizeForMatch(content)
    .split(" ")
    .filter((term) => term.length >= 5)
    .slice(0, 36);
  let best: HTMLElement | null = null;
  let bestScore = 0;
  for (const element of candidates) {
    const text = normalizeForMatch(element.textContent || "");
    const score = sourceTerms.reduce(
      (total, term) => total + (text.includes(term) ? Math.min(term.length, 18) : 0),
      0
    );
    if (score > bestScore) {
      best = element;
      bestScore = score;
    }
  }
  if (bestScore >= 18 && best) {
    if (chunkId) best.dataset.chunkId = chunkId;
    return best;
  }
  return null;
}

// ---------------------------------------------------------------------------
// DocumentViewer
// ---------------------------------------------------------------------------
interface DocumentViewerProps {
  doc: Document;
  scrollToPage?: number | null;
  scrollToHeading?: string | null;
  scrollToImageSrc?: string | null;
  highlightChunks?: ChatSourceChunk[];
  onScrolled?: () => void;
}

export const DocumentViewer = memo(function DocumentViewer({
  doc,
  scrollToPage,
  scrollToHeading,
  scrollToImageSrc,
  highlightChunks,
  onScrolled,
}: DocumentViewerProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const articleRef = useRef<HTMLElement>(null);
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [showToc, setShowToc] = useState(true);

  // ---- Fetch markdown content ----
  const { data: markdown, isLoading, error } = useQuery({
    queryKey: ["document-markdown", doc.id],
    queryFn: () => api.getText(`/documents/${doc.id}/markdown`),
    enabled: doc.status === "indexed",
    staleTime: 5 * 60 * 1000, // cache 5 min
  });

  // ---- Extract headings for TOC ----
  const headings = useMemo(
    () => (markdown ? extractHeadings(markdown) : []),
    [markdown]
  );

  // ---- Process markdown (insert page dividers) ----
  const processedMarkdown = useMemo(() => {
    return markdown ? insertPageDividers(markdown, doc.page_count) : "";
  }, [markdown, doc.page_count]);

  // ---- Stable ReactMarkdown components (prevents DOM recreation on re-render) ----
  // Without memoization, inline arrow functions create new references each render,
  // causing React to unmount/remount all heading elements — destroying highlight classes.
  const mdComponents = useMemo<import("react-markdown").Components>(() => ({
    h1: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h1 id={id} {...props}>{children}</h1>;
    },
    h2: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h2 id={id} {...props}>{children}</h2>;
    },
    h3: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h3 id={id} {...props}>{children}</h3>;
    },
    h4: ({ children, ...props }) => {
      const text = getHeadingText(children);
      const id = generateHeadingId(text);
      return <h4 id={id} {...props}>{children}</h4>;
    },
    p: ({ children, node, ...props }) => {
      const pageNo = getPageToken(children);
      if (pageNo) return <PageDivider pageNo={pageNo} />;

      const nodeChildren = (node as {
        children?: Array<{ type?: string; tagName?: string }>;
      } | undefined)?.children;
      const hasImage = nodeChildren?.some(
        (child) => child.type === "element" && child.tagName === "img"
      );
      if (hasImage)
        return (
          <div className="mb-3 leading-relaxed text-foreground/80" {...props}>
            {children}
          </div>
        );
      return <p {...props}>{children}</p>;
    },
    // A rule beyond the declared page count is document content, not a page break.
    hr: (props) => <hr {...props} />,
    img: ({ src, alt, ...props }) => (
      <figure className="my-4">
        <img
          src={src}
          alt={alt || ""}
          loading="lazy"
          className="rounded-lg max-w-full mx-auto border border-border/30"
          style={{ minHeight: 120, objectFit: "contain", background: "var(--muted)" }}
          onLoad={(e) => {
            (e.target as HTMLImageElement).style.minHeight = "auto";
            (e.target as HTMLImageElement).style.background = "none";
          }}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
          {...props}
        />
        {alt && (
          <figcaption className="text-xs text-muted-foreground text-center mt-1.5 italic">
            {alt}
          </figcaption>
        )}
      </figure>
    ),
  }), []);

  // Stable plugin arrays
  const remarkPlugins = useMemo(() => [remarkGfm, remarkMath], []);
  const rehypePlugins = useMemo(() => [rehypeKatex], []);

  // ---- Intersection observer for active heading ----
  useEffect(() => {
    if (!contentRef.current || headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveHeading(entry.target.id);
          }
        }
      },
      { root: contentRef.current, rootMargin: "-20% 0px -60% 0px", threshold: 0 }
    );

    const headingElements = contentRef.current.querySelectorAll("h1, h2, h3, h4");
    headingElements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [headings, processedMarkdown]);

  // ---- Scroll-to support (from citation cross-link) ----
  // Manual scrollTop + rAF animation — immune to browser cancelling smooth scroll
  // on layout shifts (lazy images, tab switches, etc.)

  const scrollTo = useCallback(
    (
      target: HTMLElement,
      block: "start" | "center" = "center",
      onDone?: () => void
    ) => {
      const container = contentRef.current;
      if (!container) return;

      const calcTarget = () => {
        // Calculate offset of target relative to scroll container
        let offset = 0;
        let el: HTMLElement | null = target;
        while (el && el !== container) {
          offset += el.offsetTop;
          el = el.offsetParent as HTMLElement | null;
        }
        const targetH = target.offsetHeight;
        const containerH = container.clientHeight;
        const dest =
          block === "center"
            ? offset - containerH / 2 + targetH / 2
            : offset;
        return Math.max(0, Math.min(dest, container.scrollHeight - containerH));
      };

      // Animate with rAF (cannot be cancelled by browser unlike smooth scrollIntoView)
      const animate = (dest: number) => {
        const start = container.scrollTop;
        const dist = dest - start;
        if (Math.abs(dist) < 1) return;
        const duration = Math.min(400, Math.abs(dist) * 0.5 + 150);
        const t0 = performance.now();
        const step = () => {
          const p = Math.min((performance.now() - t0) / duration, 1);
          const ease = 1 - Math.pow(1 - p, 3); // easeOutCubic
          container.scrollTop = start + dist * ease;
          if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      };

      // Execute: scroll now, then correction after images may have loaded
      animate(calcTarget());
      const correctionTimeout = setTimeout(() => {
        animate(calcTarget());
        onDone?.();
      }, 800);
      return () => clearTimeout(correctionTimeout);
    },
    []
  );

  // Ref to track cleanup for previous scroll operations
  const scrollCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Cleanup previous scroll operation
    scrollCleanupRef.current?.();
    scrollCleanupRef.current = null;

    if (!contentRef.current || !markdown) return;
    if (!scrollToImageSrc && !scrollToHeading && !scrollToPage && !highlightChunks?.length) return;

    // Double rAF: first waits for React commit, second waits for browser paint.
    // This ensures ReactMarkdown has fully rendered headings/page-dividers/images
    // before we calculate scroll positions — critical after tab switch (KG→Content).
    const rafId = requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!contentRef.current) return;

      // Image citation — scroll to exact image element
      if (scrollToImageSrc) {
        const imgEl = contentRef.current.querySelector(
          `img[src="${CSS.escape(scrollToImageSrc)}"]`
        ) as HTMLElement | null;
        if (imgEl) {
          const figure = (imgEl.closest("figure") || imgEl) as HTMLElement;
          scrollCleanupRef.current = scrollTo(figure, "center", onScrolled) ?? null;
          // Flash highlight on figure
          figure.classList.add("ring-2", "ring-primary/50", "rounded-lg", "transition-all");
          setTimeout(() => {
            figure.classList.remove("ring-2", "ring-primary/50", "rounded-lg", "transition-all");
          }, 2500);
          return;
        }
        // Fallback: scroll to page
        if (scrollToPage) {
          const pageEl = contentRef.current.querySelector(
            `[data-page="${scrollToPage}"]`
          ) as HTMLElement | null;
          if (pageEl) {
            scrollCleanupRef.current = scrollTo(pageEl, "start", onScrolled) ?? null;
          }
        }
        return;
      }

      const sourceTarget = articleRef.current && highlightChunks?.length
        ? findChunkElement(
            articleRef.current,
            highlightChunks[0].content,
            highlightChunks[0].chunk_id,
          )
        : null;
      if (sourceTarget) {
        scrollCleanupRef.current = scrollTo(sourceTarget, "center", onScrolled) ?? null;
        return;
      }

      // Page is the authoritative citation location. Heading paths are only a
      // fallback because a retrieved chunk can cross a heading boundary.
      if (scrollToPage) {
        const pageEl = contentRef.current.querySelector(
          `[data-page="${scrollToPage}"]`
        ) as HTMLElement | null;
        if (pageEl) {
          scrollCleanupRef.current = scrollTo(pageEl, "start", onScrolled) ?? null;
          return;
        }
      }

      if (scrollToHeading) {
        const targetId = scrollToHeading
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, "")
          .replace(/\s+/g, "-")
          .slice(0, 80);
        const el = contentRef.current.querySelector(
          `#${CSS.escape(targetId)}`
        ) as HTMLElement | null;
        if (el) {
          scrollCleanupRef.current = scrollTo(el, "center", onScrolled) ?? null;
          el.classList.add("bg-primary/20", "transition-colors");
          setTimeout(() => el.classList.remove("bg-primary/20"), 2000);
          return;
        }
      }

    }));

    return () => cancelAnimationFrame(rafId);
  }, [scrollToPage, scrollToHeading, scrollToImageSrc, highlightChunks, markdown, processedMarkdown, onScrolled, scrollTo]);

  // ---- Highlight chunks from citations ----
  // Depends on processedMarkdown so highlights re-apply after document switch
  // (markdown loads async → DOM not ready when highlightChunks first set)
  useEffect(() => {
    if (!contentRef.current) return;

    // Always clear previous highlights first
    contentRef.current.querySelectorAll(".chunk-hl").forEach((el) => {
      (el as HTMLElement).classList.remove(
        "chunk-hl",
        "chunk-hl-source",
        "chunk-hl-heading",
        "chunk-hl-sibling"
      );
    });

    if (!highlightChunks || highlightChunks.length === 0) return;

    for (const chunk of highlightChunks) {
      const sourceEl = articleRef.current && findChunkElement(
        articleRef.current,
        chunk.content,
        chunk.chunk_id,
      );
      if (sourceEl) {
        sourceEl.classList.add("chunk-hl", "chunk-hl-source");
        continue;
      }

      // Strategy: find the heading from heading_path, highlight it + siblings
      const lastHeading =
        chunk.heading_path.length > 0
          ? chunk.heading_path[chunk.heading_path.length - 1]
          : null;

      if (lastHeading) {
        const headingId = generateHeadingId(lastHeading);
        const headingEl = contentRef.current.querySelector(
          `#${CSS.escape(headingId)}`
        );
        if (headingEl) {
          // Highlight heading
          headingEl.classList.add(
            "chunk-hl",
            "chunk-hl-heading"
          );
          // Highlight siblings until next heading
          let sibling = headingEl.nextElementSibling;
          let count = 0;
          while (sibling && !sibling.tagName.match(/^H[1-4]$/) && count < 20) {
            sibling.classList.add(
              "chunk-hl",
              "chunk-hl-sibling"
            );
            sibling = sibling.nextElementSibling;
            count++;
          }
          continue;
        }
      }

    }
    // Scroll is handled by the scroll-to effect above — don't compete here
  }, [highlightChunks, processedMarkdown]);

  // ---- TOC heading click ----
  const handleTocSelect = useCallback((id: string) => {
    if (!contentRef.current) return;
    const el = contentRef.current.querySelector(`#${CSS.escape(id)}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveHeading(id);
    }
  }, []);

  // ---- Loading / error / empty states ----
  if (doc.status !== "indexed") {
    return <ViewerEmpty />;
  }
  if (isLoading) return <ViewerSkeleton />;
  if (error) return <ViewerError message={(error as Error).message} />;
  if (!markdown || markdown.trim().length === 0) return <ViewerEmpty />;

  return (
    <div className="flex h-full min-h-0">
      {/* TOC sidebar */}
      {showToc && (
        <TOCSidebar
          headings={headings}
          activeId={activeHeading}
          onSelect={handleTocSelect}
        />
      )}

      {/* Main markdown content */}
      <div ref={contentRef} className="flex-1 min-h-0 overflow-y-auto">
        {/* TOC toggle (for smaller screens / when TOC hidden) */}
        {headings.length > 0 && (
          <button
            onClick={() => setShowToc(!showToc)}
            className={cn(
              "sticky top-2 left-2 z-10 p-1.5 rounded-md border bg-background/80 backdrop-blur-sm",
              "hover:bg-muted transition-colors xl:hidden",
              "flex items-center gap-1 text-xs text-muted-foreground"
            )}
          >
            <List className="w-3.5 h-3.5" />
            <ChevronRight className={cn("w-3 h-3 transition-transform", showToc && "rotate-90")} />
          </button>
        )}

        <div className="px-6 py-4">
          {/* Document title header */}
          <div className="mb-4 pb-3 border-b">
            <h2 className="text-lg font-semibold">{doc.original_filename}</h2>
            <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
              {doc.page_count && doc.page_count > 0 && <span>{doc.page_count} 页</span>}
              {doc.chunk_count > 0 && <span>{doc.chunk_count} 个文段</span>}
            </div>
          </div>

          {/* Rendered markdown */}
          <article
            ref={articleRef}
            className={cn(
              "prose prose-sm max-w-none text-foreground/80",
              // Headings — explicit foreground for light/dark theme support
              "[&_h1]:text-xl [&_h1]:font-bold [&_h1]:mt-6 [&_h1]:mb-3 [&_h1]:scroll-mt-4 [&_h1]:text-foreground",
              "[&_h2]:text-lg [&_h2]:font-semibold [&_h2]:mt-5 [&_h2]:mb-2 [&_h2]:scroll-mt-4 [&_h2]:text-foreground",
              "[&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2 [&_h3]:scroll-mt-4 [&_h3]:text-foreground",
              "[&_h4]:text-sm [&_h4]:font-semibold [&_h4]:mt-3 [&_h4]:mb-1.5 [&_h4]:scroll-mt-4 [&_h4]:text-foreground",
              // Body text
              "[&_p]:text-foreground/80 [&_p]:leading-relaxed [&_p]:mb-3",
              "[&_li]:text-foreground/80",
              "[&_strong]:text-foreground",
              // Tables
              "[&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
              "[&_th]:bg-muted/50 [&_th]:border [&_th]:border-border [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-medium [&_th]:text-foreground/80",
              "[&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1.5 [&_td]:text-foreground/80",
              // Code
              "[&_code]:bg-muted/50 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:text-foreground/90",
              "[&_pre]:bg-muted/30 [&_pre]:rounded-lg [&_pre]:p-3 [&_pre]:overflow-x-auto [&_pre]:text-xs",
              // Blockquotes
              "[&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-foreground/60",
              // Images
              "[&_img]:rounded-lg [&_img]:max-w-full [&_img]:my-3",
              // Links
                "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2",
                "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-2",
                "[&_.katex]:text-[0.95em]",
                "[&_li]:text-foreground/90"
            )}
          >
              <ReactMarkdown
                remarkPlugins={remarkPlugins}
                rehypePlugins={rehypePlugins}
                components={mdComponents}
            >
              {processedMarkdown}
            </ReactMarkdown>
          </article>
        </div>
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function getHeadingText(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(getHeadingText).join("");
  if (children && typeof children === "object" && "props" in children) {
    return getHeadingText((children as React.ReactElement<{ children?: React.ReactNode }>).props.children);
  }
  return String(children ?? "");
}

function generateHeadingId(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .slice(0, 80);
}
