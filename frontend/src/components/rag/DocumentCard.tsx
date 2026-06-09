import { memo, useState, useEffect } from "react";
import {
  Trash2,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Sparkles,
  Tags,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Document, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------
const STATUS_CONFIG: Record<DocumentStatus, { label: string; className: string; icon: typeof CheckCircle2 }> = {
  pending:    { label: "待处理", className: "bg-muted text-muted-foreground",       icon: Clock },
  parsing:    { label: "解析中", className: "bg-primary/10 text-primary",           icon: Loader2 },
  indexing:   { label: "索引中", className: "bg-amber-400/15 text-amber-400",       icon: Loader2 },
  processing: { label: "处理中", className: "bg-amber-400/15 text-amber-400",       icon: Loader2 },
  indexed:    { label: "就绪", className: "bg-primary/15 text-primary",             icon: CheckCircle2 },
  failed:     { label: "失败",   className: "bg-destructive/15 text-destructive",   icon: XCircle },
};

function StatusBadge({ status }: { status: DocumentStatus }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  const Icon = config.icon;
  const isAnimated = status === "parsing" || status === "indexing" || status === "processing";

  return (
    <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full", config.className)}>
      <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />
      {config.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// DocumentCard
// ---------------------------------------------------------------------------
interface DocumentCardProps {
  doc: Document;
  selected?: boolean;
  onDelete: (id: number) => void;
  onReindex: (id: number) => void;
  onProcess: (id: number) => void;
  isProcessing?: boolean;
  onEditMetadata: (document: Document) => void;
  onClick?: (doc: Document) => void;
}

function truncateFilenameMiddle(filename: string, maxLength = 28): string {
  if (filename.length <= maxLength) return filename;

  const extensionMatch = filename.match(/\.[^./\\]+$/);
  const extension = extensionMatch?.[0] ?? "";
  const basename = extension ? filename.slice(0, -extension.length) : filename;
  const visibleLength = Math.max(6, maxLength - extension.length - 1);
  const prefixLength = Math.ceil(visibleLength / 2);
  const suffixLength = Math.floor(visibleLength / 2);

  return `${basename.slice(0, prefixLength)}…${basename.slice(-suffixLength)}${extension}`;
}

export const DocumentCard = memo(function DocumentCard({
  doc,
  selected,
  onDelete,
  onReindex,
  onProcess,
  isProcessing,
  onEditMetadata,
  onClick,
}: DocumentCardProps) {
  const sizeStr = doc.file_size >= 1024 * 1024
    ? `${(doc.file_size / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.round(doc.file_size / 1024)} KB`;

  const isActive = doc.status === "parsing" || doc.status === "indexing" || doc.status === "processing";

  // Elapsed time for active processing
  const [elapsed, setElapsed] = useState("");
  useEffect(() => {
    if (!isActive) { setElapsed(""); return; }
    // Database timestamps are UTC but are serialized without a suffix. Parse
    // a suffix-less value as UTC so a browser in China does not show an extra
    // eight hours of elapsed processing time.
    const rawTimestamp = doc.updated_at;
    const timestamp = /(?:Z|[+-]\d{2}:?\d{2})$/.test(rawTimestamp)
      ? rawTimestamp
      : `${rawTimestamp}Z`;
    const start = new Date(timestamp).getTime();
    const tick = () => {
      const sec = Math.floor((Date.now() - start) / 1000);
      if (sec < 60) setElapsed(`${sec}s`);
      else setElapsed(`${Math.floor(sec / 60)}m ${sec % 60}s`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [isActive, doc.updated_at]);

  // Flash animation when user just clicked "Analyze"
  const handleProcess = (e: React.MouseEvent) => {
    e.stopPropagation();
    onProcess(doc.id);
  };

  return (
    <div
      className={cn(
        "group relative rounded-lg border bg-card transition-colors",
        // Active processing state — animated border glow
        isActive
          ? "border-primary/50 shadow-sm"
          : "border-border hover:shadow-md",
        selected && "border-primary ring-1 ring-primary/30 shadow-sm",
        doc.status === "indexed" ? "cursor-pointer" : "cursor-default",
      )}
      onClick={() => onClick?.(doc)}
    >
      <div className="relative px-4 py-3 flex items-start gap-3">
        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate" title={doc.original_filename}>
            {truncateFilenameMiddle(doc.original_filename)}
          </p>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs text-muted-foreground">{sizeStr}</span>
            <StatusBadge status={doc.status} />
            {isActive && (
              <span className="text-xs text-primary/80 font-medium">
                 正在处理{elapsed ? `（${elapsed}）` : "..."}
              </span>
            )}
          </div>
          {doc.error_message && (
            <p className="text-xs text-destructive mt-1 truncate">{doc.error_message}</p>
          )}
          {doc.metadata_requires_reindex && (
            <p className="mt-1 text-[10px] text-amber-600">语义元数据已变更，请重新处理文档以更新检索结果。</p>
          )}
          {Object.keys(doc.custom_metadata || {}).length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {Object.entries(doc.custom_metadata).slice(0, 2).map(([key, value]) => (
                <span key={key} className="max-w-28 truncate rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground" title={Array.isArray(value) ? value.join(", ") : String(value)}>
                  {Array.isArray(value) ? value.join(", ") : String(value)}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <Button
            variant="ghost"
            size="icon"
            onClick={(event) => { event.stopPropagation(); onEditMetadata(doc); }}
            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
            title="编辑元数据"
          >
            <Tags className="w-3.5 h-3.5" />
          </Button>
          {/* Analyze button — visible for pending/failed documents */}
          {(doc.status === "pending" || doc.status === "failed") && (
            <Button
              variant="default"
              size="sm"
              onClick={handleProcess}
              disabled={isProcessing}
              className="h-7 text-xs gap-1.5"
            >
              {isProcessing ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Sparkles className="w-3 h-3" />
              )}
               处理
            </Button>
          )}
          {/* Re-process for indexed docs — hover only */}
          {doc.status === "indexed" && (
            <Button
              variant="ghost"
              size="icon"
              onClick={(e) => { e.stopPropagation(); onReindex(doc.id); }}
              className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
               title="重新处理"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </Button>
          )}
          {/* Delete — hover only */}
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => { e.stopPropagation(); onDelete(doc.id); }}
            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Trash2 className="w-3.5 h-3.5 text-destructive" />
          </Button>
        </div>
      </div>
    </div>
  );
});
