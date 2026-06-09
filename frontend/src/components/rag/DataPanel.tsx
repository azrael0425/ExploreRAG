import { useState, useMemo, useCallback, memo } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import {
  ArrowLeft,
  FileText,
  Pencil,
  Check,
  X,
  Loader2,
  Sparkles,
  Database,
  Settings2,
  BarChart3,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UploadZone } from "./UploadZone";
import { DocumentFilters, type FilterStatus } from "./DocumentFilters";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { DocumentCard } from "./DocumentCard";
import { DocumentMetadataDialog } from "./DocumentMetadataDialog";
import { WorkspaceSettings } from "./WorkspaceSettings";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useWorkspaces } from "@/hooks/useWorkspaces";
import type { Document, DocumentStatus, KnowledgeBase, UpdateWorkspace } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>([
  "parsing",
  "indexing",
  "processing",
]);
const PROCESSABLE_STATUSES = new Set<DocumentStatus>(["pending", "failed"]);

interface DataPanelProps {
  workspace: KnowledgeBase | undefined;
  documents: Document[] | undefined;
  docsLoading: boolean;
  selectedDocId: number | null;
  onSelectDoc: (doc: Document) => void;
  onUpload: (files: File[], metadata: Record<string, unknown>) => Promise<void>;
  isUploading: boolean;
  onUpdateMetadata: (document: Document, metadata: Record<string, unknown>) => Promise<void>;
  onDelete: (id: number) => void;
  onProcess: (id: number) => void;
  onReindex: (id: number) => void;
  isProcessing: boolean;
  onUpdateWorkspace: (data: UpdateWorkspace) => Promise<void>;
}

export const DataPanel = memo(function DataPanel({
  workspace,
  documents,
  docsLoading,
  selectedDocId,
  onSelectDoc,
  onUpload,
  isUploading,
  onUpdateMetadata,
  onDelete,
  onProcess,
  onReindex,
  isProcessing,
  onUpdateWorkspace,
}: DataPanelProps) {
  const navigate = useNavigate();
  const { data: workspaces } = useWorkspaces();
  const [deleteDocConfirm, setDeleteDocConfirm] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<FilterStatus>("all");
  const [isEditingName, setIsEditingName] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [importFiles, setImportFiles] = useState<File[]>([]);
  const [metadataDoc, setMetadataDoc] = useState<Document | null>(null);
  const handleFilesSelected = useCallback((files: File[]) => {
    setImportFiles(files);
  }, []);
  const handleImport = useCallback(async (metadata: Record<string, unknown>) => {
    await onUpload(importFiles, metadata);
    setImportFiles([]);
  }, [importFiles, onUpload]);
  const handleMetadataUpdate = useCallback(async (metadata: Record<string, unknown>) => {
    if (!metadataDoc) return;
    await onUpdateMetadata(metadataDoc, metadata);
    setMetadataDoc(null);
  }, [metadataDoc, onUpdateMetadata]);

  const processingCount = useMemo(
    () => documents?.filter((d) => PROCESSING_STATUSES.has(d.status)).length ?? 0,
    [documents]
  );

  const pendingCount = useMemo(
    () => documents?.filter((d) => PROCESSABLE_STATUSES.has(d.status)).length ?? 0,
    [documents]
  );

  const filteredDocs = useMemo(() => {
    if (!documents) return [];
    let result = documents;
    if (statusFilter !== "all") {
      if (statusFilter === "parsing") {
        result = result.filter((d) => PROCESSING_STATUSES.has(d.status));
      } else {
        result = result.filter((d) => d.status === statusFilter);
      }
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter((d) =>
        d.original_filename.toLowerCase().includes(q)
      );
    }
    return result;
  }, [documents, statusFilter, searchQuery]);

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = { all: 0 };
    documents?.forEach((d) => {
      counts.all = (counts.all || 0) + 1;
      counts[d.status] = (counts[d.status] || 0) + 1;
    });
    return counts as Record<FilterStatus, number>;
  }, [documents]);

  const handleBatchProcess = useCallback(async () => {
    if (!documents || batchProcessing) return;
    const processable = documents.filter((d) => PROCESSABLE_STATUSES.has(d.status));
    if (processable.length === 0) return;

    setBatchProcessing(true);
    const count = processable.length;
    toast.info(`正在处理 ${count} 个文档...`, {
      description: "文档将按顺序处理。",
    });

    try {
      await api.post("/rag/process-batch", {
        document_ids: processable.map((d) => d.id),
      });
    } catch {
      toast.error("无法启动批量处理");
    } finally {
      setBatchProcessing(false);
    }
  }, [documents, batchProcessing]);

  const handleStartEdit = () => {
    if (workspace) {
      setEditName(workspace.name);
      setEditDesc(workspace.description || "");
      setIsEditingName(true);
    }
  };

  const handleSaveEdit = async () => {
    if (!editName.trim()) return;
    await onUpdateWorkspace({
      name: editName.trim(),
      description: editDesc.trim() || undefined,
    });
    setIsEditingName(false);
  };

  return (
    <div className="relative h-full flex flex-col overflow-hidden">
      {/* Workspace summary and switcher */}
      <div className="flex-shrink-0 border-b bg-background px-3 py-3">
        <button
          onClick={() => navigate("/")}
          className="flex h-7 items-center gap-1.5 rounded-md px-1.5 text-[11px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <ArrowLeft className="w-3 h-3" />
          知识库
        </button>

        <div className="mt-2 rounded-lg border bg-muted/20 p-1.5">
          <p className="px-1 pb-1 text-[10px] font-semibold text-muted-foreground">
            知识库列表
          </p>
          <div className="max-h-28 overflow-y-auto space-y-0.5 pr-0.5">
            {workspaces?.map((item) => {
              const isCurrent = item.id === workspace?.id;
              return (
                <button
                  key={item.id}
                  onClick={() => !isCurrent && navigate(`/knowledge-bases/${item.id}`)}
                  className={cn(
                    "flex h-7 w-full items-center gap-1.5 rounded-md px-1.5 text-left text-xs transition-colors",
                    isCurrent
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  )}
                >
                  <Database className="w-3 h-3 flex-shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{item.name}</span>
                  <span className="text-[10px] tabular-nums text-muted-foreground/70">
                    {item.document_count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {isEditingName ? (
          <div className="mt-2 space-y-1.5 rounded-lg border bg-background p-2">
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveEdit()}
              placeholder="名称"
              autoFocus
              className="text-sm font-semibold h-8"
            />
            <Input
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              placeholder="描述"
              className="text-xs h-7"
            />
            <div className="flex items-center gap-1">
              <Button size="sm" onClick={handleSaveEdit} disabled={!editName.trim()} className="h-6 text-[10px] px-2">
                <Check className="w-3 h-3 mr-0.5" /> 保存
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setIsEditingName(false)} className="h-6 text-[10px] px-2">
                <X className="w-3 h-3 mr-0.5" /> 取消
              </Button>
            </div>
          </div>
        ) : (
          <div className="mt-2 flex items-center gap-1.5 rounded-lg border bg-background p-2">
            <div className="flex-1 min-w-0">
              <h1 className="text-sm font-bold truncate">
                {workspace?.name || "知识库"}
              </h1>
              {workspace?.description && (
                <p className="text-[10px] text-muted-foreground truncate">
                  {workspace.description}
                </p>
              )}
            </div>
            <Button
              title="RAG 评测"
              size="icon"
              variant="ghost"
              onClick={() => workspace && navigate(`/knowledge-bases/${workspace.id}/evaluation`)}
              className="h-7 w-7 flex-shrink-0"
            >
              <BarChart3 className="w-3.5 h-3.5" />
            </Button>
            <Button
              title="知识库设置"
              size="icon"
              variant="ghost"
              onClick={() => setSettingsOpen(true)}
              className="h-7 w-7 flex-shrink-0"
            >
              <Settings2 className="w-3.5 h-3.5" />
            </Button>
            <Button
              title="编辑知识库名称"
              size="icon"
              variant="ghost"
              onClick={handleStartEdit}
              className="h-7 w-7 flex-shrink-0"
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
          </div>
        )}
      </div>

      {/* Upload */}
      <div className="flex-shrink-0 border-b bg-background px-3 py-3">
        <UploadZone onFilesSelected={handleFilesSelected} isUploading={isUploading} mini />
      </div>

      {/* Document section */}
      <div className="flex-shrink-0 space-y-1.5 border-b bg-background px-3 py-3">
        <div className="flex items-center">
          <h2 className="text-xs font-semibold flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5" />
            文档
          </h2>
        </div>

        {/* Analyze All banner — compact for narrow panel */}
        {pendingCount > 0 && (
          <button
            onClick={handleBatchProcess}
            disabled={batchProcessing || processingCount > 0}
            className={cn(
              "w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-md",
              "border border-primary/20 bg-primary/[0.06]",
              "hover:bg-primary/10 transition-colors",
              (batchProcessing || processingCount > 0) && "opacity-50 pointer-events-none",
            )}
          >
            <div className="flex items-center gap-2 min-w-0">
              <Sparkles className={cn("w-3.5 h-3.5 text-primary flex-shrink-0", batchProcessing && "animate-spin")} />
              <span className="text-[11px] font-medium text-primary truncate">
                {batchProcessing ? "正在启动..." : `处理全部（${pendingCount}）`}
              </span>
            </div>
            <span className="text-[10px] text-muted-foreground flex-shrink-0">
              {pendingCount} 个待处理
            </span>
          </button>
        )}
      </div>

      {/* Document list — ~80% */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {docsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground mr-2" />
            <span className="text-xs text-muted-foreground">加载中...</span>
          </div>
        ) : !documents || documents.length === 0 ? (
          <div className="flex-1 flex items-center justify-center px-3">
            <p className="text-xs text-muted-foreground text-center">
              暂无文档。请在上方拖放或上传文件。
            </p>
          </div>
        ) : (
          <>
            <div className="px-3 pt-2 flex-shrink-0">
              <DocumentFilters
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                statusFilter={statusFilter}
                onStatusChange={setStatusFilter}
                counts={statusCounts}
              />
            </div>

            <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
              {filteredDocs.map((doc) => (
                <DocumentCard
                  key={doc.id}
                  doc={doc}
                  selected={doc.id === selectedDocId}
                  onDelete={setDeleteDocConfirm}
                  onReindex={onReindex}
                  onProcess={onProcess}
                  isProcessing={isProcessing}
                  onEditMetadata={setMetadataDoc}
                  onClick={onSelectDoc}
                />
              ))}
              {filteredDocs.length === 0 && documents.length > 0 && (
                <div className="text-center py-4 text-[11px] text-muted-foreground">
                  没有符合筛选条件的文档
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={deleteDocConfirm !== null}
        onConfirm={async () => {
          if (deleteDocConfirm !== null) {
            onDelete(deleteDocConfirm);
            setDeleteDocConfirm(null);
          }
        }}
        onCancel={() => setDeleteDocConfirm(null)}
        title="删除文档"
        message="确定删除吗？这会移除该文档及其索引数据。"
        confirmLabel="删除"
        variant="danger"
      />

      {workspace && (
        <WorkspaceSettings
          workspace={workspace}
          onSave={onUpdateWorkspace}
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
        />
      )}
      {workspace && (
        <DocumentMetadataDialog
          open={importFiles.length > 0}
          title="导入文档元数据"
          schema={workspace.metadata_schema}
          files={importFiles}
          saving={isUploading}
          onClose={() => setImportFiles([])}
          onSave={handleImport}
        />
      )}
      {workspace && metadataDoc && (
        <DocumentMetadataDialog
          open
          title={`编辑元数据：${metadataDoc.original_filename}`}
          schema={workspace.metadata_schema}
          initialValues={metadataDoc.custom_metadata}
          saving={isUploading}
          onClose={() => setMetadataDoc(null)}
          onSave={handleMetadataUpdate}
        />
      )}
    </div>
  );
});
