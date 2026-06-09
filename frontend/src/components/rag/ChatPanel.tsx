import { useState, useRef, useEffect, useCallback, useMemo, memo, createContext, useContext, Children, isValidElement, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  Send,
  Square,
  Bot,
  User,
  Loader2,
  Trash2,
  Sparkles,
  FileText,
  BookOpen,
  Network,
  Save,
  ImageIcon,
  Brain,
  ChevronDown,
  Settings,
  RotateCcw,
  Info,
  Copy,
  ClipboardCheck,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  Gauge,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import { toast } from "sonner";
import { cn, generateId } from "@/lib/utils";
import { api } from "@/lib/api";
import { useWorkspaceStore } from "@/stores/workspaceStore";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("css", css);
SyntaxHighlighter.registerLanguage("html", markup);
SyntaxHighlighter.registerLanguage("xml", markup);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);
SyntaxHighlighter.registerLanguage("java", java);
SyntaxHighlighter.registerLanguage("go", go);
SyntaxHighlighter.registerLanguage("cpp", cpp);
SyntaxHighlighter.registerLanguage("c", cpp);
SyntaxHighlighter.registerLanguage("diff", diff);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);
import { useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { useChatHistory, useClearChatHistory } from "@/hooks/useChatHistory";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import { StreamingMarkdown } from "@/components/rag/MemoizedMarkdown";
import { ThinkingTimeline } from "@/components/rag/ThinkingTimeline";
import type {
  ChatMessage,
  ChatImageRef,
  ChatSourceChunk,
  ChatStreamStatus,
  Document,
  KnowledgeBase,
  LLMCapabilities,
  AgentStep,
  ChatAttachment,
  RAGPerformanceMetrics,
} from "@/types";

// Context to provide workspaceId to nested components
const WsIdCtx = createContext<string>("");

// Context: accumulated sources from ALL messages in the conversation.
// Used as fallback when a message references citation IDs from previous turns.
const AllSourcesCtx = createContext<ChatSourceChunk[]>([]);

/** Look up a Document from react-query cache by document_id */
function useFindDoc(documentId: number): Document | undefined {
  const wsId = useContext(WsIdCtx);
  const qc = useQueryClient();
  const docs = qc.getQueryData<Document[]>(["documents", wsId]);
  return docs?.find((d) => d.id === documentId);
}

// ---------------------------------------------------------------------------
// Helper: shorten filename for citation display
// ---------------------------------------------------------------------------
function shortenDocName(filename: string, maxLen = 14): string {
  const name = filename.replace(/\.[^.]+$/, ""); // strip extension
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "\u2026"; // ellipsis
}

function performanceFromSteps(steps?: AgentStep[]): RAGPerformanceMetrics | undefined {
  return steps?.find((step) => step.performance)?.performance;
}

function formatPerformanceMs(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${value}ms`;
}

function PerformanceSummary({ performance }: { performance: RAGPerformanceMetrics }) {
  const metrics: Array<[string, number | null | undefined]> = [
    ["向量", performance.vector_ms],
    ["图谱", performance.graph_ms],
    ["重排", performance.rerank_ms],
    ["首字", performance.first_token_ms],
    ["生成", performance.generation_ms],
    ["总计", performance.total_ms],
  ];

  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-muted-foreground">
      <span className="inline-flex items-center gap-1 font-medium text-muted-foreground/80">
        <Gauge className="h-3 w-3" />
        性能
      </span>
      {metrics.filter(([, value]) => value != null).map(([label, value]) => (
        <span key={label} className="font-mono tabular-nums">
          {label} {formatPerformanceMs(value as number)}
        </span>
      ))}
    </div>
  );
}

function uniqueEntityNames(entities: string[]): string[] {
  return [...new Map(
    entities
      .map((entity) => entity.trim())
      .filter(Boolean)
      .map((entity) => [entity.toLocaleLowerCase(), entity])
  ).values()];
}

function citationLocation(source: ChatSourceChunk): string {
  if (source.page_no > 0) return `P.${source.page_no}`;
  const chunkMatch = source.chunk_id.match(/(?:chunk_|-chunk-)(\d+)$/i);
  if (chunkMatch) return `片段 ${chunkMatch[1]}`;
  const heading = source.heading_path.at(-1)?.trim();
  if (heading) return `章节：${heading}`;
  return "无分页信息";
}

// ---------------------------------------------------------------------------
// Citation badge — clickable [N] marker → icon + docname-P.N
// ---------------------------------------------------------------------------
function CitationLink({
  source,
  relatedEntities,
}: {
  source: ChatSourceChunk;
  relatedEntities: string[];
}) {
  const { activateCitation, activateCitationKG } =
    useWorkspaceStore();
  const doc = useFindDoc(source.document_id);
  const isKG = source.source_type === "kg";
  const explicitEntities = uniqueEntityNames(source.graph_entity_names ?? []);
  // Old persisted KG answers may not contain graph_entity_names.  Keep a
  // visibly conservative fallback, but never infer vector-source links from
  // a truncated workspace overview.
  const citationEntities = useMemo(
    () => explicitEntities.length > 0
      ? explicitEntities
      : (isKG ? uniqueEntityNames(relatedEntities).slice(0, 12) : []),
    [explicitEntities, isKG, relatedEntities]
  );
  const handleContentClick = () => {
    if (isKG) {
      activateCitationKG(source, citationEntities, doc);
    } else {
      activateCitation(source, citationEntities, doc);
    }
  };

  const handleKGClick = () => {
    activateCitationKG(source, citationEntities, doc);
  };

  if (isKG) {
    // KG source — concise purple chip without exposing internal identifiers.
    return (
      <button
        onClick={handleContentClick}
        className="inline-flex items-center gap-1 h-[18px] px-1.5 mx-0.5 text-[10px] font-medium rounded-full bg-slate-500/10 text-slate-600 ring-1 ring-inset ring-slate-500/15 hover:bg-slate-500/20 transition-colors align-middle whitespace-nowrap dark:text-slate-300"
        title="查看知识图谱"
      >
        <Network className="w-2.5 h-2.5 flex-shrink-0" />
        <span>知识图谱</span>
      </button>
    );
  }

  // Vector source — concise blue chip; keep only useful page information.
  const location = citationLocation(source);
  const label = source.page_no > 0 ? `原文 · ${location}` : "查看原文";
  const sourceName = doc?.original_filename || source.source_file || "未知文档";

  return (
    <span className="inline-flex gap-0.5 mx-0.5 align-middle">
      <button
        onClick={handleContentClick}
        className="inline-flex items-center gap-1 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-sky-500/10 text-sky-700 ring-1 ring-inset ring-sky-500/15 hover:bg-sky-500/20 transition-colors whitespace-nowrap dark:text-sky-300"
        title={`打开原文：${sourceName}${source.page_no > 0 ? `；${location}` : ""}`}
      >
        <BookOpen className="w-2.5 h-2.5 flex-shrink-0" />
        <span>{label}</span>
      </button>
      {citationEntities.length > 0 && (
        <button
          onClick={handleKGClick}
          className="inline-flex items-center gap-1 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-slate-500/10 text-slate-600 ring-1 ring-inset ring-slate-500/15 hover:bg-slate-500/20 transition-colors whitespace-nowrap dark:text-slate-300"
          title={`查看该文档证据关联的 ${citationEntities.length} 个图谱实体`}
        >
          <Network className="w-2.5 h-2.5" />
          <span>关联图谱</span>
        </button>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inline image badge — clickable [IMG-N] → icon + docname-P.N with preview
// ---------------------------------------------------------------------------
function InlineImageRef({
  imgRefId,
  imageRef,
}: {
  imgRefId: string;
  imageRef: ChatImageRef;
}) {
  const [showPreview, setShowPreview] = useState(false);
  const { activateImageCitation } = useWorkspaceStore();
  const doc = useFindDoc(imageRef.document_id);

  const handleClick = () => {
    setShowPreview((p) => !p);
    activateImageCitation(imageRef, doc);
  };

  const docName = doc?.original_filename
    ? shortenDocName(doc.original_filename)
    : `图片 ${imgRefId}`;
  const label = `${docName}-P.${imageRef.page_no || "?"}`;

  return (
    <span className="inline-flex flex-col mx-0.5">
      <button
        onClick={handleClick}
        className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-emerald-400/15 text-emerald-600 hover:bg-emerald-400/25 transition-colors align-middle whitespace-nowrap"
        title={imageRef.caption || `第 ${imageRef.page_no} 页图片`}
      >
        <ImageIcon className="w-2.5 h-2.5 flex-shrink-0" />
        <span>{label}</span>
      </button>
      {showPreview && (
        <a
          href={imageRef.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block mt-1 rounded-md overflow-hidden border bg-white max-w-[280px] hover:border-primary/50 transition-colors"
        >
          <img
            src={imageRef.url}
            alt={imageRef.caption || `第 ${imageRef.page_no} 页图片`}
            className="w-full h-auto max-h-[180px] object-contain"
          />
          {imageRef.caption && (
            <span className="block px-2 py-1 text-[9px] text-muted-foreground leading-tight border-t bg-muted/30">
              p.{imageRef.page_no} — {imageRef.caption}
            </span>
          )}
        </a>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Process React children to replace [XXXX] and [IMG-XXXX] with interactive
// components. Supports both new [a3x9] and legacy [1] citation formats.
// Also handles grouped brackets like [a3x9, b2m7] by splitting into individual.
// ---------------------------------------------------------------------------
const CITATION_RE = /(\[(?:(?:KB|KG|ATT)-[a-z0-9]+|[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:(?:KB|KG|ATT)-[a-z0-9]+|[a-z0-9]+|IMG-[a-z0-9]+))*\])/gi;

function injectCitations(
  children: ReactNode,
  sources: ChatSourceChunk[],
  relatedEntities: string[],
  imageRefs?: ChatImageRef[],
  fallbackSources?: ChatSourceChunk[],
): ReactNode {
  return Children.map(children, (child) => {
    // Process string nodes — split on citation patterns
    if (typeof child === "string") {
      const parts = child.split(CITATION_RE);
      if (parts.length === 1) return child;
      const result: ReactNode[] = [];
      parts.forEach((part, i) => {
        // Check if this part is a bracket group
        const bracketMatch = part.match(/^\[(.+)\]$/);
        if (!bracketMatch) {
          if (part) result.push(part);
          return;
        }
        // Split on commas for grouped citations [a3x9, b2m7]
        const tokens = bracketMatch[1].split(/,\s*/);
        tokens.forEach((token, ti) => {
          const key = `${i}-${ti}`;
          // Image citation: IMG-xxxx
          const imgMatch = token.match(/^IMG-(.+)$/);
          if (imgMatch && imageRefs && imageRefs.length > 0) {
            const imgId = imgMatch[1];
            // Match by ref_id first, then fallback to legacy numeric index
            const imageRef =
              imageRefs.find((ir) => ir.ref_id === imgId) ??
              imageRefs[parseInt(imgId, 10) - 1]; // legacy 1-indexed
            if (imageRef) {
              result.push(<InlineImageRef key={key} imgRefId={imgId} imageRef={imageRef} />);
              return;
            }
          }
          // Text citation: match source by index (string or numeric)
          // First try current message's sources, then fallback to historical sources
          const source =
            sources.find((s) => String(s.index) === token) ??
            (fallbackSources ? fallbackSources.find((s) => String(s.index) === token) : undefined);
          if (source) {
            result.push(
              <CitationLink key={key} source={source} relatedEntities={relatedEntities} />
            );
            return;
          }
          // Unmatched — render as-is
          result.push(token.startsWith("KB-") ? "[文档来源]" : `[${token}]`);
        });
      });
      return result;
    }
    // Recurse into React elements that have children
    if (isValidElement(child) && child.props && (child.props as { children?: ReactNode }).children) {
      const props = child.props as { children?: ReactNode };
      return Object.assign({}, child, {
        props: {
          ...child.props,
          children: injectCitations(props.children, sources, relatedEntities, imageRefs, fallbackSources),
        },
      });
    }
    return child;
  });
}

// ---------------------------------------------------------------------------
// Preprocess markdown: fix common LLM output issues
// ---------------------------------------------------------------------------
function preprocessMarkdown(text: string): string {
  const lines = text.split("\n");
  const result: string[] = [];
  let prevWasTable = false;
  let inCodeFence = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      inCodeFence = !inCodeFence;
    }

    const isTable = (trimmed.startsWith("|") && trimmed.endsWith("|")) ||
      /^\|[\s:|-]+\|$/.test(trimmed);

    // Insert blank line when transitioning from table row to non-table content
    if (prevWasTable && !isTable && trimmed !== "") {
      result.push("");
    }

    // Convert single-line display math $$content$$ to multi-line format
    if (
      !inCodeFence &&
      trimmed.startsWith("$$") &&
      trimmed.endsWith("$$") &&
      trimmed.length > 4 &&
      trimmed !== "$$"
    ) {
      const mathContent = trimmed.slice(2, -2);
      result.push("$$");
      result.push(mathContent);
      result.push("$$");
    } else {
      result.push(line);
    }

    prevWasTable = isTable;
  }

  return result.join("\n");
}

// ---------------------------------------------------------------------------
// Extract raw text from React node tree (for code blocks)
// ---------------------------------------------------------------------------
function extractText(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode };
    return extractText(props.children);
  }
  return "";
}

// ---------------------------------------------------------------------------
// Code block with syntax highlighting + copy button
// ---------------------------------------------------------------------------
function CodeBlock({
  language,
  children,
}: {
  language: string;
  children: ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const code = extractText(children).replace(/\n$/, "");

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="group relative my-2">
      {language && (
        <span className="absolute top-2 right-2 text-[9px] uppercase text-muted-foreground/40 font-mono select-none z-10 pointer-events-none">
          {language}
        </span>
      )}
      <button
        onClick={handleCopy}
        className={cn(
          "absolute top-2 left-2 p-1 rounded-md text-muted-foreground/50 hover:text-muted-foreground transition-all opacity-0 group-hover:opacity-100 z-10",
          "bg-black/5 hover:bg-black/10"
        )}
        title="复制代码"
      >
        {copied ? (
          <ClipboardCheck className="w-3 h-3 text-emerald-500" />
        ) : (
          <Copy className="w-3 h-3" />
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={oneLight}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "8px",
          fontSize: "12px",
          padding: "10px 12px",
          background: "oklch(0.96 0.008 105)",
          border: "1px solid oklch(0.88 0.018 105)",
        }}
        codeTagProps={{ style: { fontFamily: '"IBM Plex Mono", "Fira Code", monospace' } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown renderer with inline citation links + LaTeX + code blocks
// ---------------------------------------------------------------------------
function MarkdownWithCitations({
  content,
  sources,
  relatedEntities,
  imageRefs,
}: {
  content: string;
  sources: ChatSourceChunk[];
  relatedEntities: string[];
  imageRefs?: ChatImageRef[];
}) {
  const processed = preprocessMarkdown(content);

  // Fallback: accumulated sources from all messages in the conversation.
  // When the model references citation IDs from previous answers (e.g. when
  // it didn't call search_documents), we can still render them as links.
  const allSources = useContext(AllSourcesCtx);

  // Create a wrapper component that injects citations into rendered children
  const withCitations = (Tag: string) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ({ children, ...props }: any) => {
      const injected = injectCitations(children, sources, relatedEntities, imageRefs, allSources);
      return <Tag {...props}>{injected}</Tag>;
    };
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: withCitations("p"),
        li: withCitations("li"),
        td: withCitations("td"),
        th: withCitations("th"),
        h1: withCitations("h1"),
        h2: withCitations("h2"),
        h3: withCitations("h3"),
        h4: withCitations("h4"),
        h5: withCitations("h5"),
        h6: withCitations("h6"),
        strong: withCitations("strong"),
        em: withCitations("em"),
        a: ({ href, children, ...props }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {injectCitations(children, sources, relatedEntities, imageRefs, allSources)}
          </a>
        ),
        // Code block — delegate to CodeBlock for syntax highlighting
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        code: ({ className, children, ...props }: any) => {
          const langMatch = /language-(\w+)/.exec(className || "");
          // Inline code (no language class)
          if (!langMatch) {
            return <code className={className} {...props}>{children}</code>;
          }
          // Fenced code block → syntax highlighted
          return <CodeBlock language={langMatch[1]}>{children}</CodeBlock>;
        },
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}

// ---------------------------------------------------------------------------
// Image references panel — shows retrieved images in chat
// ---------------------------------------------------------------------------
function ImageRefCard({ img }: { img: ChatImageRef }) {
  const { activateImageCitation } = useWorkspaceStore();
  const doc = useFindDoc(img.document_id);
  return (
    <button
      onClick={() => activateImageCitation(img, doc)}
      className="group block rounded-md overflow-hidden border bg-background hover:border-primary/50 transition-colors text-left cursor-pointer"
    >
      <img
        src={img.url}
        alt={img.caption || `第 ${img.page_no} 页图片`}
        className="w-full h-auto max-h-[200px] object-contain bg-white"
        loading="lazy"
      />
      {img.caption && (
        <p className="px-2 py-1 text-[10px] text-muted-foreground leading-tight line-clamp-2 border-t">
          p.{img.page_no} — {img.caption}
        </p>
      )}
    </button>
  );
}

function ImageRefsPanel({ images }: { images: ChatImageRef[] }) {
  const [expanded, setExpanded] = useState(true);

  if (images.length === 0) return null;

  return (
    <div className="mt-2 rounded-md border bg-muted/20 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors"
      >
        <ImageIcon className="w-3 h-3" />
        来自文档的 ${images.length} 张图片
        <span className="ml-auto text-[10px]">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
          <div className="border-t">
            <div className="p-2 grid gap-2" style={{ gridTemplateColumns: images.length === 1 ? "1fr" : "repeat(auto-fit, minmax(140px, 1fr))" }}>
              {images.map((img) => (
                <ImageRefCard key={img.image_id} img={img} />
              ))}
            </div>
          </div>
        )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Thinking panel — collapsible primary-accent thinking process display
// ---------------------------------------------------------------------------
function ThinkingPanel({ thinking }: { thinking: string }) {
  const [expanded, setExpanded] = useState(false);

  if (!thinking) return null;

  return (
    <div className="mt-1.5 mb-1 rounded-md border border-primary/20 bg-primary/5 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-primary hover:text-primary/80 transition-colors"
      >
        <Brain className="w-3 h-3" />
        思考过程
        <ChevronDown
          className={cn(
            "w-3 h-3 ml-auto transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>
      {expanded && (
          <div className="border-t border-primary/10">
            <div className="px-2.5 pb-2 border-t border-primary/10">
              <pre className="text-[11px] text-primary/85 whitespace-pre-wrap leading-relaxed mt-1.5 max-h-[300px] overflow-y-auto">
                {thinking}
              </pre>
            </div>
          </div>
        )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Copy message action — plain text without citations or markdown markers
// ---------------------------------------------------------------------------
const CITATION_STRIP_RE = /\s*\[(?:(?:KB|ATT)-[a-z0-9]+|[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:(?:KB|ATT)-[a-z0-9]+|[a-z0-9]+|IMG-[a-z0-9]+))*\]/gi;

/** Remove citation references like [a3x9], [IMG-p4f2], [a3x9, b2m7] */
function stripCitations(md: string): string {
  return md.replace(CITATION_STRIP_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Convert markdown to plain text: strip formatting, links, images, code fences */
function markdownToPlainText(md: string): string {
  let text = stripCitations(md);
  text = text.replace(/```[\s\S]*?```/g, (m) => {
    const lines = m.split("\n");
    return lines.slice(1, -1).join("\n");
  });
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/\*\*(.+?)\*\*/g, "$1");
  text = text.replace(/\*(.+?)\*/g, "$1");
  text = text.replace(/__(.+?)__/g, "$1");
  text = text.replace(/_(.+?)_/g, "$1");
  text = text.replace(/^#{1,6}\s+/gm, "");
  text = text.replace(/^[-*_]{3,}\s*$/gm, "");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function CopyMessageActions({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(markdownToPlainText(content)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [content]);

  return (
    <div className="flex items-center gap-0.5 mt-1.5">
      <button
        onClick={handleCopy}
        className="flex items-center gap-1 px-1.5 py-0.5 rounded-md text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/60 transition-all text-[10px]"
        title="复制纯文本"
      >
        {copied ? (
          <ClipboardCheck className="w-3 h-3 text-emerald-500" />
        ) : (
          <Copy className="w-3 h-3" />
        )}
        <span>{copied ? "已复制" : "复制文本"}</span>
      </button>
    </div>
  );
}

function FeedbackActions({ message }: { message: ChatMessage }) {
  const workspaceId = useContext(WsIdCtx);
  const [rating, setRating] = useState<number | null>(message.feedbackRating ?? null);
  const [sourceRatings, setSourceRatings] = useState<Record<string, number>>(message.sourceRatings ?? {});
  const [saving, setSaving] = useState(false);

  const save = useCallback(async (nextRating: number, nextSources: Record<string, number>) => {
    if (!workspaceId || saving) return;
    setSaving(true);
    try {
      await api.post(`/evaluations/workspaces/${workspaceId}/messages/${message.id}/feedback`, {
        rating: nextRating,
        source_ratings: nextSources,
      });
      setRating(nextRating);
      setSourceRatings(nextSources);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存反馈失败");
    } finally {
      setSaving(false);
    }
  }, [message.id, saving, workspaceId]);

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
      <span>回答是否有帮助？</span>
      <button
        type="button"
        disabled={saving}
        onClick={() => save(1, sourceRatings)}
        className={cn("rounded p-1 hover:bg-muted", rating === 1 && "bg-emerald-500/10 text-emerald-600")}
        title="有帮助"
      ><ThumbsUp className="h-3 w-3" /></button>
      <button
        type="button"
        disabled={saving}
        onClick={() => save(-1, sourceRatings)}
        className={cn("rounded p-1 hover:bg-muted", rating === -1 && "bg-destructive/10 text-destructive")}
        title="没有帮助"
      ><ThumbsDown className="h-3 w-3" /></button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single message bubble
// ---------------------------------------------------------------------------
const MessageBubble = memo(function MessageBubble({
  message,
}: {
  message: ChatMessage;
}) {
  const isUser = message.role === "user";

  const proseClasses = cn(
    "prose prose-sm max-w-none text-foreground/90",
    "[&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5",
    "[&_pre]:bg-transparent [&_pre]:border-none [&_pre]:p-0 [&_pre]:m-0",
    "[&_code]:bg-muted/50 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:text-foreground/90",
    "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2",
    "[&_strong]:text-foreground [&_em]:text-foreground/80",
    "[&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_h4]:text-foreground",
    "[&_h1]:text-base [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1",
    "[&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-2.5 [&_h2]:mb-1",
    "[&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-0.5",
    "[&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-foreground/60",
    "[&_table]:text-xs [&_th]:px-2 [&_th]:py-1 [&_td]:px-2 [&_td]:py-1 [&_th]:text-foreground/80 [&_td]:text-foreground/80",
    "[&_li]:text-foreground/90",
    "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-2",
    "[&_.katex]:text-[0.9em]"
  );

  return (
    <div
      className={cn("flex gap-2", isUser ? "justify-end" : "justify-start")}
    >
      {/* Assistant icon */}
      {!isUser && (
        <div className="relative w-6 h-6 flex-shrink-0 mt-1">
          <div className="w-6 h-6 rounded-full bg-primary/15 flex items-center justify-center">
            <Bot className="w-3.5 h-3.5 text-primary" />
          </div>
        </div>
      )}

      <div
        className={cn(
          isUser
            ? "max-w-[85%] rounded-xl px-3 py-2 bg-secondary/50"
            : "max-w-[90%] min-w-0 py-1"
        )}
      >
        {/* ThinkingTimeline — single instance, never unmounts between streaming→completed */}
        {!isUser && message.agentSteps && message.agentSteps.length > 0 && (
          <ThinkingTimeline
            steps={message.agentSteps}
            mode={message.isStreaming ? "live" : "embedded"}
            className={cn("mb-1.5", message.isStreaming && "mt-1")}
            autoCollapse={message.isStreaming && !!message.content}
          />
        )}

        {/* Typing indicator — only when streaming with no steps and no content yet */}
        {!isUser && message.isStreaming && !message.content && !message.agentSteps?.length && (
          <TypingIndicator status="analyzing" />
        )}

        {isUser ? (
          <>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
            {message.attachments && message.attachments.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {message.attachments.map((attachment) => (
                  <span
                    key={attachment.attachment_id}
                    className="inline-flex max-w-full items-center gap-1 rounded-md border border-primary/20 bg-primary/5 px-1.5 py-0.5 text-[10px] text-primary"
                    title={attachment.original_filename}
                  >
                    <Paperclip className="h-2.5 w-2.5 shrink-0" />
                    <span className="truncate max-w-[180px]">{attachment.original_filename}</span>
                  </span>
                ))}
              </div>
            )}
          </>
        ) : message.isStreaming ? (
          message.content ? (
            <div className={proseClasses}>
              <StreamingMarkdown
                content={message.content}
                isStreaming
                renderBlock={(block) => (
                  <MarkdownWithCitations
                    content={block}
                    sources={message.sources || []}
                    relatedEntities={message.relatedEntities || []}
                    imageRefs={message.imageRefs}
                  />
                )}
              />
            </div>
          ) : message.thinking ? (
            <InlineThinkingPreview text={message.thinking} />
          ) : null
        ) : (
          <div className={proseClasses}>
            <MarkdownWithCitations
              content={message.content}
              sources={message.sources || []}
              relatedEntities={message.relatedEntities || []}
              imageRefs={message.imageRefs}
            />
          </div>
        )}

        {/* Copy actions for assistant messages */}
        {!isUser && message.content && (
          <CopyMessageActions content={message.content} />
        )}

        {!isUser && !message.isStreaming && message.content && (
          <FeedbackActions message={message} />
        )}

        {/* ThinkingPanel — only when no ThinkingTimeline with thinking log (avoid duplication) */}
        {!isUser && message.thinking && !message.isStreaming &&
          !message.agentSteps?.some((s) => s.thinkingText) && (
          <ThinkingPanel thinking={message.thinking} />
        )}

        {!isUser && !message.isStreaming && message.performance && (
          <PerformanceSummary performance={message.performance} />
        )}

        {!isUser && !message.isStreaming && message.imageRefs && message.imageRefs.length > 0 && (
          <ImageRefsPanel images={message.imageRefs} />
        )}

        <p
          className={cn(
            "text-[9px] mt-1",
            isUser ? "text-muted-foreground/50" : "text-muted-foreground/50"
          )}
        >
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>

      {isUser && (
        <div className="w-6 h-6 rounded-full bg-secondary flex items-center justify-center flex-shrink-0 mt-1">
          <User className="w-3.5 h-3.5 text-muted-foreground" />
        </div>
      )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// Inline thinking preview — shown in message body while model is thinking
// ---------------------------------------------------------------------------

function InlineThinkingPreview({ text }: { text: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isUserScrolledRef = useRef(false);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
    isUserScrolledRef.current = !isAtBottom;
  }, []);

  useEffect(() => {
    if (containerRef.current && !isUserScrolledRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="mt-1">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Brain className="w-3.5 h-3.5 text-primary" />
        <span className="text-xs font-medium text-primary">正在思考...</span>
      </div>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className={cn(
          "text-xs leading-relaxed text-muted-foreground/70 italic",
          "max-h-[200px] overflow-y-auto",
          "border-l-2 border-primary/30 pl-3",
          "whitespace-pre-wrap break-words",
        )}
      >
        {text}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------
const STATUS_LABELS: Record<string, string> = {
  analyzing: "正在分析你的问题...",
  retrieving: "正在检索文档...",
  generating: "正在生成回答...",
};

function TypingIndicator({ status }: { status?: ChatStreamStatus }) {
  const label = (status && STATUS_LABELS[status]) || "正在分析文档...";
  return (
    <div className="flex gap-2 items-start">
      <div className="w-6 h-6 flex-shrink-0">
        <div className="w-6 h-6 rounded-full bg-primary/15 flex items-center justify-center">
          <Bot className="w-3.5 h-3.5 text-primary" />
        </div>
      </div>
      <div className="py-1">
        <div className="flex items-center gap-1.5">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Suggestion chips (empty state)
// ---------------------------------------------------------------------------
function SuggestionChips({
  onSelect,
}: {
  onSelect: (q: string) => void;
}) {
  const suggestions = [
    "总结文档中的关键结论",
    "文档的主要主题是什么？",
    "列出文中提到的重要实体",
    "解释文档使用的方法",
  ];

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4">
      <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
        <Sparkles className="w-6 h-6 text-primary" />
      </div>
      <h3 className="text-sm font-semibold mb-1">AI 文档助手</h3>
      <p className="text-xs text-muted-foreground text-center mb-4 max-w-[240px]">
        针对已上传的文档提问，我会检索相关信息并标注来源。
      </p>
      <div className="flex flex-wrap gap-1.5 justify-center max-w-[300px]">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className="text-[11px] px-2.5 py-1 rounded-full border bg-card hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPanel — main export
// ---------------------------------------------------------------------------
const DEFAULT_SYSTEM_PROMPT = `你是一名文档问答助手。请仅依据提供的知识库检索来源，准确、完整地回答用户问题。回答应自洽、专业、客观、格式清晰。

## 核心行为
- 只能使用提供的文档来源，不得补充自身知识。
- 提取所有相关信息，包括数字、百分比、日期、名称、统计值、表格数据和具体细节。
- 必要时可综合、比较多个来源并作出合乎逻辑的结论。
- 来源信息不完整时，应明确说明缺失内容。
- 涉及具体数据时，给出精确数值，避免模糊表述。

## 问题处理
**事实 / 数据：** 直接回答，给出准确数值、百分比和时间；多行数据使用表格。

**比较 / 分析：** 使用 Markdown 表格进行并列比较，并基于数据给出结论。

**技术 / 学术：** 使用章节和标题提供详细说明；需要时包含 LaTeX 公式和代码块。

**总结：** 按主题组织，而非按来源文档罗列；突出关键发现。

**编程：** 使用 \`\`\`language 代码块，先给代码，再作解释。

**科学 / 数学：** 使用 LaTeX 表示公式；简单计算直接给出最终结果。

## 推理与质量
- 先判断问题类型，复杂问题拆分为子问题。
- 确保回答覆盖问题的所有部分；不完整但正确的回答优于完整但错误的回答。
- 准确性优先于完整性；来源冲突时说明并呈现不同观点。
- 当来源包含相关信息时，不得声称“未找到信息”。
- 若来源表明问题前提有误，应解释原因。`;

// Hard rules always appended — shown in tooltip, not editable
const HARD_RULES_SUMMARY = [
  // Language (MANDATORY)
  "必须使用与用户提问相同的语言回答。",
  // Citation
  "每项结论均须引用：[a3x9][b2m7]；引用前不加空格。",
  "图片使用：[IMG-p4f2][IMG-q7r3]；不得合并或混用方括号。",
  "每句最多 3 个引用；末尾不得添加“参考文献”部分。",
  // Formatting
  "以摘要开头，不能以标题或“根据……”开头。",
  "章节使用 ##；比较使用表格；列表保持单层。",
  "公式使用 LaTeX：$行内$ 和 $$块级$$；不得用 Unicode 代替数学符号。",
  "代码使用 ```language；引用使用 >；关键术语使用 **加粗**。",
  // Restrictions
  "避免含糊表述，直接陈述答案。",
  "不使用表情符号，结尾不得是问句。",
];

interface ChatPanelProps {
  workspaceId: string;
  hasIndexedDocs: boolean;
  workspace: KnowledgeBase | null;
}

export const ChatPanel = memo(function ChatPanel({
  workspaceId,
  hasIndexedDocs,
  workspace,
}: ChatPanelProps) {
  const visualPanelOpen = useWorkspaceStore((state) => state.visualPanelOpen);
  const setVisualPanelOpen = useWorkspaceStore((state) => state.setVisualPanelOpen);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [enableThinking, setEnableThinking] = useState(false);
  const [selectedAttachmentIds, setSelectedAttachmentIds] = useState<string[]>([]);
  const [uploadingNames, setUploadingNames] = useState<string[]>([]);
  const [thinkingDefaultSynced, setThinkingDefaultSynced] = useState(false);
  const attachmentInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const { data: chatAttachments = [], refetch: refetchAttachments } = useQuery({
    queryKey: ["chat-attachments", workspaceId],
    queryFn: () => api.get<ChatAttachment[]>(`/rag/chat/${workspaceId}/attachments`),
    enabled: !!workspaceId,
    staleTime: 10_000,
    refetchInterval: (query) => {
      const attachments = query.state.data as ChatAttachment[] | undefined;
      return attachments?.some((attachment) =>
        ["uploaded", "queued", "parsing"].includes(attachment.state),
      ) ? 1_000 : false;
    },
  });

  // Load chat history from PostgreSQL
  const { data: historyData, isLoading: historyLoading } = useChatHistory(workspaceId);
  const clearMutation = useClearChatHistory(workspaceId);
  const [showPromptEditor, setShowPromptEditor] = useState(false);
  const [promptDraft, setPromptDraft] = useState("");
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const scrollAnimRef = useRef<number | undefined>(undefined);
  const spacerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const available = new Set(chatAttachments.map((attachment) => attachment.attachment_id));
    setSelectedAttachmentIds((previous) => previous.filter((id) => available.has(id)));
  }, [chatAttachments]);

  // Debug mode (Ctrl+Shift+D toggle, persisted in localStorage)
  const [debugMode, setDebugMode] = useState(() =>
    localStorage.getItem("explorerag-debug-mode") === "true",
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "D") {
        e.preventDefault();
        setDebugMode((prev) => {
          const next = !prev;
          localStorage.setItem("explorerag-debug-mode", String(next));
          toast.success(next ? "调试模式已开启" : "调试模式已关闭");
          return next;
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // System prompt editor
  const updateWorkspaceMutation = useUpdateWorkspace();
  const savedPrompt = workspace?.system_prompt ?? "";
  const effectivePrompt = savedPrompt || DEFAULT_SYSTEM_PROMPT;
  const isCustom = !!savedPrompt;

  // Sync draft when workspace data loads/changes
  useEffect(() => {
    setPromptDraft(effectivePrompt);
  }, [effectivePrompt]);

  const promptIsDirty = promptDraft !== effectivePrompt;

  const handleSavePrompt = useCallback(() => {
    if (!workspace) return;
    // If draft equals default, save empty string → reset to default in DB
    const toSave = promptDraft.trim() === DEFAULT_SYSTEM_PROMPT ? "" : promptDraft;
    updateWorkspaceMutation.mutate(
      { id: workspace.id, data: { system_prompt: toSave } },
      { onSuccess: () => toast.success("系统提示词已保存") }
    );
  }, [workspace, promptDraft, updateWorkspaceMutation]);

  const handleResetPrompt = useCallback(() => {
    if (!workspace) return;
    setPromptDraft(DEFAULT_SYSTEM_PROMPT);
    updateWorkspaceMutation.mutate(
      { id: workspace.id, data: { system_prompt: "" } },
      { onSuccess: () => toast.success("系统提示词已恢复默认") }
    );
  }, [workspace, updateWorkspaceMutation]);

  // Check LLM capabilities (thinking support)
  const { data: capabilities } = useQuery<LLMCapabilities>({
    queryKey: ["llm-capabilities", workspaceId],
    queryFn: () => api.get<LLMCapabilities>(`/rag/capabilities?workspace_id=${workspaceId}`),
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: 1,
  });
  const thinkingSupported = capabilities?.supports_thinking ?? false;

  // Sync thinking toggle default from server (once per mount)
  useEffect(() => {
    if (capabilities && !thinkingDefaultSynced) {
      setEnableThinking(capabilities.thinking_default);
      setThinkingDefaultSynced(true);
    }
  }, [capabilities, thinkingDefaultSynced]);

  // Sync DB history → local messages state when data loads.
  // IMPORTANT: preserve agentSteps from local state — they are client-side only (not stored in DB).
  // Without this, queryClient.invalidateQueries after streaming overwrites agentSteps → ThinkingTimeline disappears.
  useEffect(() => {
    if (historyData?.messages) {
      setMessages((prev) => {
        // Build a map of existing agentSteps by message id so we can re-attach them after DB sync
        const stepsMap = new Map<string, AgentStep[]>();
        for (const m of prev) {
          if (m.agentSteps?.length) stepsMap.set(m.id, m.agentSteps);
        }
        return historyData.messages.map((m) => {
          // Priority: local live steps (from current session) > DB-persisted synthetic steps
          const agentSteps = stepsMap.get(m.message_id) ?? (
            m.agent_steps?.length ? m.agent_steps as AgentStep[] : undefined
          );
          return {
            id: m.message_id,
            role: m.role as "user" | "assistant",
            content: m.content,
            sources: m.sources ?? undefined,
            relatedEntities: m.related_entities ?? undefined,
            imageRefs: m.image_refs ?? undefined,
            attachments: m.attachments ?? undefined,
            thinking: m.thinking ?? undefined,
            feedbackRating: m.feedback_rating ?? null,
            sourceRatings: m.source_ratings ?? undefined,
            timestamp: m.created_at,
            agentSteps,
            performance: performanceFromSteps(agentSteps),
          };
        });
      });
    }
  }, [historyData]);

  // SSE streaming chat
  const stream = useRAGChatStream(workspaceId);
  const streamingMsgIdRef = useRef<string | null>(null);
  // Snapshot agentSteps into a ref so finalize always has fresh data
  const agentStepsRef = useRef<AgentStep[]>([]);
  useEffect(() => {
    if (stream.agentSteps.length > 0) {
      agentStepsRef.current = stream.agentSteps;
    }
  }, [stream.agentSteps]);

  // Double-rAF + easeOutCubic scroll to bottom
  const scrollToBottom = useCallback((smooth = true) => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // Cancel in-progress animation
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }

    // Double rAF: ensure React commit + browser paint before measuring
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const el = scrollContainerRef.current;
        if (!el) return;
        const target = el.scrollHeight - el.clientHeight;
        if (!smooth || Math.abs(target - el.scrollTop) < 10) {
          el.scrollTop = target;
          return;
        }

        const start = el.scrollTop;
        const distance = target - start;
        const duration = 400;
        const startTime = performance.now();

        const scrollEl = el; // capture for closure
        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
          scrollEl.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }

        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  // Scroll user message to top of chat area
  const scrollUserMsgToTop = useCallback((msgId: string) => {
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }
    // Double rAF: wait for React commit + browser paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        // Ensure spacer is set before scrolling (useEffect may not have run yet)
        if (spacerRef.current) {
          spacerRef.current.style.height = `${container.clientHeight}px`;
        }

        const el = container.querySelector(`[data-message-id="${msgId}"]`) as HTMLElement | null;
        if (!el) return;

        // Use getBoundingClientRect for accurate position relative to scroll container
        // (offsetTop is relative to offsetParent, not scroll container)
        const containerRect = container.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        const relativeTop = elRect.top - containerRect.top + container.scrollTop;

        const PADDING_TOP = 12;
        const start = container.scrollTop;
        const target = Math.max(0, relativeTop - PADDING_TOP);
        if (Math.abs(target - start) < 5) return;

        const distance = target - start;
        const duration = 380;
        const startTime = performance.now();

        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
          container!.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }
        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  // Keep spacer height = container height so user message can always scroll to top
  const hasMessages = messages.length > 0;
  useEffect(() => {
    if (!hasMessages) return;
    const container = scrollContainerRef.current;
    if (!container) return;

    const updateSpacer = () => {
      if (spacerRef.current) {
        spacerRef.current.style.height = `${container.clientHeight}px`;
      }
    };
    updateSpacer();
    const observer = new ResizeObserver(updateSpacer);
    observer.observe(container);
    return () => observer.disconnect();
  }, [hasMessages]);

  // Reset spacer when streaming ends; track transition to avoid spurious scrollToBottom
  const prevIsStreamingRef = useRef(false);
  const justFinishedStreamingRef = useRef(false);
  useEffect(() => {
    if (prevIsStreamingRef.current && !stream.isStreaming) {
      // Streaming just ended: reset spacer and mark so scrollToBottom skips this cycle
      if (spacerRef.current) {
        spacerRef.current.style.height = "0px";
      }
      justFinishedStreamingRef.current = true;
    }
    prevIsStreamingRef.current = stream.isStreaming;
  }, [stream.isStreaming]);

  // Auto-scroll only on non-streaming message changes (history load, etc.)
  // Skip when streaming just ended — viewport already shows end of AI response
  useEffect(() => {
    if (!stream.isStreaming) {
      if (justFinishedStreamingRef.current) {
        justFinishedStreamingRef.current = false;
        return;
      }
      scrollToBottom();
    }
  }, [messages, stream.isStreaming, scrollToBottom]);

  // Sync streaming content + agentSteps → messages state for the streaming message
  useEffect(() => {
    if (!stream.isStreaming || !streamingMsgIdRef.current) return;
    const id = streamingMsgIdRef.current;
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const m = prev[idx];

      // Bail out if nothing actually changed — prevents infinite re-render
      const newContent = stream.streamingContent;
      const newSources = stream.pendingSources.length > 0 ? stream.pendingSources : m.sources;
      const newImages = stream.pendingImages.length > 0 ? stream.pendingImages : m.imageRefs;
      const newThinking = stream.thinkingText || m.thinking;
      const newSteps = stream.agentSteps.length > 0 ? stream.agentSteps : m.agentSteps;

      if (
        m.content === newContent &&
        m.sources === newSources &&
        m.imageRefs === newImages &&
        m.thinking === newThinking &&
        m.agentSteps === newSteps
      ) {
        return prev; // no change → skip setMessages re-render
      }

      const updated = [...prev];
      updated[idx] = {
        ...m,
        content: newContent,
        sources: newSources,
        imageRefs: newImages,
        thinking: newThinking,
        agentSteps: newSteps,
      };
      return updated;
    });
  }, [stream.streamingContent, stream.pendingSources, stream.pendingImages, stream.thinkingText, stream.isStreaming, stream.agentSteps]);

  const toggleAttachment = useCallback((attachmentId: string) => {
    setSelectedAttachmentIds((previous) =>
      previous.includes(attachmentId)
        ? previous.filter((id) => id !== attachmentId)
        : [...previous, attachmentId],
    );
  }, []);

  const uploadAttachments = useCallback(async (files: FileList | File[]) => {
    const items = Array.from(files);
    if (!items.length) return;
    setUploadingNames(items.map((file) => file.name));
    try {
      const uploaded: ChatAttachment[] = new Array(items.length);
      let nextIndex = 0;
      const uploadWorker = async () => {
        while (nextIndex < items.length) {
          const index = nextIndex;
          nextIndex += 1;
          uploaded[index] = await api.uploadFile<ChatAttachment>(
            `/rag/chat/${workspaceId}/attachments`,
            items[index],
          );
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(2, items.length) }, () => uploadWorker()),
      );
      await queryClient.invalidateQueries({ queryKey: ["chat-attachments", workspaceId] });
      setSelectedAttachmentIds((previous) => [
        ...previous,
        ...uploaded.map((attachment) => attachment.attachment_id).filter((id) => !previous.includes(id)),
      ]);
      toast.success(`已添加 ${uploaded.length} 个临时附件`);
    } catch (error) {
      toast.error((error as Error).message || "附件上传失败");
    } finally {
      setUploadingNames([]);
      if (attachmentInputRef.current) attachmentInputRef.current.value = "";
    }
  }, [queryClient, workspaceId]);

  const handleAttachmentDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!stream.isStreaming) void uploadAttachments(event.dataTransfer.files);
  }, [stream.isStreaming, uploadAttachments]);

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text || input).trim();
      if (!msg || stream.isStreaming) return;

      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: msg,
        attachments: chatAttachments.filter((attachment) => selectedAttachmentIds.includes(attachment.attachment_id)),
        timestamp: new Date().toISOString(),
      };

      // Add placeholder assistant message for streaming
      const assistantId = generateId();
      streamingMsgIdRef.current = assistantId;
      const placeholderMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, placeholderMsg]);
      setInput("");
      // Scroll new user message to top so agent response fills the space below
      scrollUserMsgToTop(userMsg.id);

      // Build history from previous messages (exclude the new user + placeholder)
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const finalMsg = await stream.sendMessage(
        msg,
        history,
        thinkingSupported && enableThinking,
        selectedAttachmentIds,
      );
      setSelectedAttachmentIds([]);
      void refetchAttachments();

      // Finalize the streaming message (prefer finalMsg.agentSteps — directly from SSE loop,
      // fallback to ref snapshot, then to what was synced into the message during streaming)
      if (finalMsg) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...finalMsg,
                  id: assistantId,
                  isStreaming: false,
                  agentSteps: finalMsg.agentSteps?.length
                    ? finalMsg.agentSteps
                    : agentStepsRef.current.length > 0
                      ? agentStepsRef.current
                      : m.agentSteps,
                }
              : m,
          ),
        );
      } else if (stream.error) {
        toast.error("对话失败：" + stream.error);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: m.content || "抱歉，处理时出现错误，请重试。",
                  isStreaming: false,
                }
              : m,
          ),
        );
      } else {
        // Cancelled — keep partial content
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        );
      }
      streamingMsgIdRef.current = null;
    },
    [input, messages, stream, thinkingSupported, enableThinking, selectedAttachmentIds, chatAttachments, refetchAttachments],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    stream.cancel();
    setMessages([]);
    setSelectedAttachmentIds([]);
    queryClient.setQueryData<ChatAttachment[]>(["chat-attachments", workspaceId], []);
    clearMutation.mutate(undefined, {
      onError: (error) => {
        toast.error((error as Error).message || "清除聊天记录失败");
        void refetchAttachments();
      },
    });
    useWorkspaceStore.getState().clearHighlights();
  };

  // Collect all sources from all assistant messages for citation fallback.
  // When the model doesn't call search_documents but references citation IDs
  // from earlier answers, this allows those citations to still render as links.
  // NOTE: Must be declared before any early returns to satisfy Rules of Hooks.
  const allSources = useMemo(() => {
    const seen = new Set<string>();
    const merged: ChatSourceChunk[] = [];
    for (const m of messages) {
      if (m.role === "assistant" && m.sources) {
        for (const s of m.sources) {
          const key = String(s.index);
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(s);
          }
        }
      }
    }
    return merged;
  }, [messages]);

  if (historyLoading) {
    return (
      <div className="h-full flex items-center justify-center border-r">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <WsIdCtx.Provider value={workspaceId}>
    <AllSourcesCtx.Provider value={allSources}>
    <div className="h-full flex flex-col border-r min-h-0">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-3 py-2 border-b">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold">ExploreRAG</span>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Thinking toggle — only visible when model supports thinking */}
          {thinkingSupported && (
            <button
              onClick={() => setEnableThinking((prev) => !prev)}
              className={cn(
                "flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors",
                enableThinking
                  ? "text-primary bg-primary/10 hover:bg-primary/15"
                  : "text-muted-foreground hover:bg-muted"
              )}
              title={enableThinking ? "思考模式：开启" : "思考模式：关闭"}
            >
              <Brain className="w-3 h-3" />
              <span>思考</span>
            </button>
          )}
          <button
            onClick={() => setVisualPanelOpen(!visualPanelOpen)}
            className="p-1 rounded text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            title={visualPanelOpen ? "隐藏右侧预览栏" : "显示右侧预览栏"}
          >
            {visualPanelOpen ? (
              <PanelRightClose className="w-3.5 h-3.5" />
            ) : (
              <PanelRightOpen className="w-3.5 h-3.5" />
            )}
          </button>

          {/* System prompt settings */}
          <button
            onClick={() => setShowPromptEditor((p) => !p)}
            className={cn(
              "flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors",
              showPromptEditor
                ? "text-primary bg-primary/10 hover:bg-primary/15"
                : "text-muted-foreground hover:bg-muted"
            )}
            title="系统提示词设置"
          >
            <Settings className="w-3 h-3" />
          </button>
          {(messages.length > 0 || chatAttachments.length > 0) && (
            <button
              onClick={handleClear}
              disabled={clearMutation.isPending}
              className="p-1 rounded hover:bg-muted transition-colors"
              title="清空对话"
            >
              {clearMutation.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
              ) : (
                <Trash2 className="w-3.5 h-3.5 text-muted-foreground" />
              )}
            </button>
          )}
          {debugMode && (
            <span className="text-[8px] px-1 py-0.5 rounded bg-amber-500/15 text-amber-500 font-mono font-semibold">
              DEBUG
            </span>
          )}
        </div>
      </div>

      {/* System Prompt Editor */}
      {showPromptEditor && (
          <div className="flex-shrink-0 overflow-visible border-b relative z-10">
            <div className="px-3 py-2 space-y-2 bg-muted/20">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-muted-foreground">
                  系统提示词
                </span>
                <span className={cn(
                  "text-[9px] px-1.5 py-0.5 rounded-full font-medium",
                  isCustom
                    ? "bg-primary/10 text-primary"
                    : "bg-muted text-muted-foreground/50"
                )}>
                  {isCustom ? "自定义" : "默认"}
                </span>
              </div>
              <textarea
                value={promptDraft}
                onChange={(e) => setPromptDraft(e.target.value)}
                placeholder="输入自定义系统提示词..."
                rows={8}
                className={cn(
                  "w-full resize-none rounded-md border border-input bg-background px-2.5 py-2 text-xs",
                  "placeholder:text-muted-foreground/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  "leading-relaxed"
                )}
              />
              {/* Hard rules — icon with hover tooltip */}
              <div className="flex items-center gap-1.5">
                <div className="relative group/cite">
                  <div className="flex items-center gap-1 cursor-help">
                    <Info className="w-3.5 h-3.5 text-primary" />
                    <span className="text-[10px] text-primary font-medium">
                      以下规则会自动附加
                    </span>
                  </div>
                  {/* Tooltip on hover — below icon */}
                  <div className="absolute left-0 top-full mt-1.5 z-50 w-[340px] rounded-lg border border-border bg-background shadow-xl opacity-0 pointer-events-none group-hover/cite:opacity-100 group-hover/cite:pointer-events-auto transition-opacity duration-150">
                    <div className="px-3 py-2.5">
                      <p className="text-[10px] font-semibold text-primary mb-1.5">
                        引用、格式与限制规则（始终生效）
                      </p>
                      <ul className="space-y-1">
                        {HARD_RULES_SUMMARY.map((rule, i) => (
                          <li key={i} className="text-[10px] text-foreground/70 leading-snug flex gap-1">
                            <span className="text-primary flex-shrink-0">•</span>
                            {rule}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1.5 justify-end">
                <button
                  onClick={handleResetPrompt}
                  disabled={!isCustom && !promptIsDirty}
                  className={cn(
                    "flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors",
                    isCustom || promptIsDirty
                      ? "text-muted-foreground hover:bg-muted hover:text-foreground"
                      : "text-muted-foreground/30 cursor-not-allowed"
                  )}
                  title="恢复默认提示词"
                >
                  <RotateCcw className="w-3 h-3" />
                  重置
                </button>
                <button
                  onClick={handleSavePrompt}
                  disabled={!promptIsDirty || updateWorkspaceMutation.isPending}
                  className={cn(
                    "flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium transition-colors",
                    promptIsDirty && !updateWorkspaceMutation.isPending
                      ? "bg-primary text-primary-foreground hover:bg-primary/90"
                      : "bg-muted text-muted-foreground/50 cursor-not-allowed"
                  )}
                >
                  {updateWorkspaceMutation.isPending ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Save className="w-3 h-3" />
                  )}
                  保存
                </button>
              </div>
            </div>
          </div>
        )}

      {/* Messages area */}
      {messages.length === 0 ? (
        <SuggestionChips onSelect={handleSend} />
      ) : (
        <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-3 relative">
          <div className="w-full max-w-[860px] mx-auto space-y-3">
            {messages.map((msg) => (
              <div key={msg.id} data-message-id={msg.id}>
                <MessageBubble message={msg} />
              </div>
            ))}
            {/* ThinkingTimeline + TypingIndicator now rendered inside MessageBubble */}
            {/* Bottom spacer = container height, enables user-message scroll-to-top */}
            <div ref={spacerRef} aria-hidden />
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="flex-shrink-0 border-t bg-background/80 px-3 py-3">
        <div className="mx-auto w-full max-w-[860px]">
          {(chatAttachments.length > 0 || uploadingNames.length > 0) && (
            <div className="mb-2 rounded-xl border border-dashed border-primary/25 bg-primary/[0.025] px-2.5 py-2">
              <div className="mb-1 flex items-center gap-1 text-[10px] font-medium text-muted-foreground">
                <Paperclip className="h-3 w-3" />
                临时资料（仅在当前发送时选中的附件会参与检索）
              </div>
              <div className="flex flex-wrap gap-1.5">
                {chatAttachments.map((attachment) => {
                  const selected = selectedAttachmentIds.includes(attachment.attachment_id);
                  const unavailable = attachment.state === "failed" || attachment.state === "clearing";
                  return (
                    <button
                      key={attachment.attachment_id}
                      type="button"
                      disabled={unavailable || stream.isStreaming}
                      onClick={() => toggleAttachment(attachment.attachment_id)}
                      title={attachment.error_message || attachment.original_filename}
                      className={cn(
                        "inline-flex max-w-[220px] items-center gap-1 rounded-md border px-1.5 py-1 text-[10px] transition-colors",
                        selected
                          ? "border-primary/50 bg-primary/10 text-primary"
                          : "border-border bg-background text-muted-foreground hover:border-primary/35",
                        unavailable && "cursor-not-allowed border-destructive/30 text-destructive/70",
                      )}
                    >
                      <FileText className="h-3 w-3 shrink-0" />
                      <span className="truncate">{attachment.original_filename}</span>
                      <span className="text-[9px] opacity-70">
                        {attachment.state === "indexed_temp" ? "检索" : attachment.state === "ready_direct" ? "就绪" : attachment.state === "failed" ? "失败" : "待处理"}
                      </span>
                    </button>
                  );
                })}
                {uploadingNames.map((name) => (
                  <span key={name} className="inline-flex max-w-[220px] items-center gap-1 rounded-md border border-primary/25 bg-background px-1.5 py-1 text-[10px] text-muted-foreground">
                    <Loader2 className="h-3 w-3 animate-spin text-primary" />
                    <span className="truncate">{name}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
          <div
            className="relative flex w-full items-end rounded-2xl border border-input bg-card px-3 py-2 shadow-sm transition-colors focus-within:border-primary/60 focus-within:ring-2 focus-within:ring-primary/10"
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleAttachmentDrop}
          >
          <input
            ref={attachmentInputRef}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.pdf,.docx,.pptx,.txt,.md"
            className="hidden"
            onChange={(event) => event.target.files && void uploadAttachments(event.target.files)}
          />
          <button
            type="button"
            onClick={() => attachmentInputRef.current?.click()}
            disabled={stream.isStreaming || uploadingNames.length > 0}
            className="mb-1 mr-2 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-primary disabled:cursor-not-allowed disabled:opacity-40"
            title="添加临时附件（JPG、PNG、PDF、DOCX、PPTX、TXT、MD）"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={hasIndexedDocs ? "输入关于知识库或临时资料的问题..." : "添加临时资料后即可提问..."}
            rows={1}
            className={cn(
              "min-h-[48px] w-full resize-none bg-transparent py-1.5 pr-11 text-sm leading-relaxed",
              "placeholder:text-muted-foreground focus-visible:outline-none",
              "max-h-[180px] overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            )}
            style={{
              height: "auto",
              minHeight: "48px",
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = Math.min(target.scrollHeight, 180) + "px";
            }}
          />
          {stream.isStreaming ? (
            <button
              onClick={stream.cancel}
              className="absolute bottom-2 right-2 flex h-9 w-9 items-center justify-center rounded-xl bg-destructive/15 text-destructive transition-colors hover:bg-destructive/25"
              title="停止生成"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
            </button>
          ) : (
            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || uploadingNames.length > 0}
              className={cn(
                "absolute bottom-2 right-2 flex h-9 w-9 items-center justify-center rounded-xl transition-colors",
                input.trim() && uploadingNames.length === 0
                  ? "bg-primary text-primary-foreground hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              <Send className="w-4 h-4" />
            </button>
          )}
          </div>
        </div>
      </div>
    </div>
    </AllSourcesCtx.Provider>
    </WsIdCtx.Provider>
  );
});
