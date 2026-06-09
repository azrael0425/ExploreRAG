import { useMemo, useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DataPanel } from "@/components/rag/DataPanel";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { VisualPanel } from "@/components/rag/VisualPanel";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useWorkspace, useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { api } from "@/lib/api";
import type { Document, RAGStats, DocumentStatus, UpdateWorkspace } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>([
  "parsing",
  "indexing",
  "processing",
]);

const MIN_LEFT_COLUMN = 240;
const MIN_CHAT_COLUMN = 320;
const MIN_VISUAL_COLUMN = 320;
const RESIZER_WIDTH = 6;

type ResizeEdge = "left" | "chat";

interface ResizeState {
  edge: ResizeEdge;
  startX: number;
  startLeft: number;
  startChat: number;
  visualPanelOpen: boolean;
}

export function WorkspacePage() {
  const { workspaceId } = useParams<{ workspaceId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const wsId = workspaceId ? Number(workspaceId) : null;
  const hasValidWorkspaceId = wsId !== null && Number.isInteger(wsId) && wsId > 0;

  // -- Workspace data --
  const {
    data: workspace,
    isLoading: workspaceLoading,
    error: workspaceError,
  } = useWorkspace(hasValidWorkspaceId ? wsId : null);
  const updateWorkspace = useUpdateWorkspace();
  const {
    selectedDoc,
    selectDoc,
    reset: resetStore,
    visualPanelOpen,
  } = useWorkspaceStore();
  const layoutRef = useRef<HTMLDivElement>(null);
  const visualPanelRef = useRef<HTMLDivElement>(null);
  const lastVisualWidthRef = useRef<number | null>(null);
  const wasVisualPanelOpenRef = useRef(visualPanelOpen);
  const isVisualPanelOpening = visualPanelOpen && !wasVisualPanelOpenRef.current;
  const [columnWidths, setColumnWidths] = useState({ left: 300, chat: 520 });
  const [resizeState, setResizeState] = useState<ResizeState | null>(null);

  useEffect(() => {
    if (!visualPanelOpen || !visualPanelRef.current) return;

    const observer = new ResizeObserver(([entry]) => {
      if (!isVisualPanelOpening && entry.contentRect.width >= MIN_VISUAL_COLUMN) {
        lastVisualWidthRef.current = entry.contentRect.width;
      }
    });
    observer.observe(visualPanelRef.current);

    return () => observer.disconnect();
  }, [visualPanelOpen, isVisualPanelOpening]);

  useEffect(() => {
    const wasOpen = wasVisualPanelOpenRef.current;
    wasVisualPanelOpenRef.current = visualPanelOpen;
    if (!visualPanelOpen || wasOpen) return;

    const containerWidth = layoutRef.current?.clientWidth;
    if (!containerWidth) return;

    const availableWidth = containerWidth - columnWidths.left - RESIZER_WIDTH * 2;
    const maxVisualWidth = Math.max(MIN_VISUAL_COLUMN, availableWidth - MIN_CHAT_COLUMN);
    const preferredVisualWidth = lastVisualWidthRef.current ?? availableWidth / 2;
    const visualWidth = Math.min(
      Math.max(preferredVisualWidth, MIN_VISUAL_COLUMN),
      maxVisualWidth,
    );

    setColumnWidths((previous) => ({
      ...previous,
      chat: Math.max(MIN_CHAT_COLUMN, availableWidth - visualWidth),
    }));
  }, [visualPanelOpen, columnWidths.left]);

  const startResize = useCallback(
    (edge: ResizeEdge, event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      setResizeState({
        edge,
        startX: event.clientX,
        startLeft: columnWidths.left,
        startChat: columnWidths.chat,
        visualPanelOpen,
      });
    },
    [columnWidths, visualPanelOpen],
  );

  useEffect(() => {
    if (!resizeState) return;

    const handlePointerMove = (event: PointerEvent) => {
      const containerWidth = layoutRef.current?.clientWidth;
      if (!containerWidth) return;

      const offset = event.clientX - resizeState.startX;
      setColumnWidths((previous) => {
        if (resizeState.edge === "left") {
          const maxLeft = Math.max(
            MIN_LEFT_COLUMN,
            resizeState.visualPanelOpen
              ? containerWidth - resizeState.startChat - MIN_VISUAL_COLUMN
              : containerWidth - MIN_CHAT_COLUMN,
          );
          return {
            ...previous,
            left: Math.min(Math.max(resizeState.startLeft + offset, MIN_LEFT_COLUMN), maxLeft),
          };
        }

        const maxChat = Math.max(
          MIN_CHAT_COLUMN,
          containerWidth - resizeState.startLeft - MIN_VISUAL_COLUMN,
        );
        return {
          ...previous,
          chat: Math.min(Math.max(resizeState.startChat + offset, MIN_CHAT_COLUMN), maxChat),
        };
      });
    };

    const stopResize = () => setResizeState(null);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize, { once: true });

    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
    };
  }, [resizeState]);

  // Reset store when switching between workspaces
  useEffect(() => {
    resetStore();
  }, [workspaceId, resetStore]);

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------
  const { data: documents, isLoading: docsLoading } = useQuery({
    queryKey: ["documents", workspaceId],
    queryFn: () =>
      api.get<Document[]>(`/documents/workspace/${workspaceId}`),
    enabled: hasValidWorkspaceId,
    refetchInterval: (query) => {
      const docs = query.state.data;
      if (docs?.some((d) => PROCESSING_STATUSES.has(d.status))) return 3000;
      return false;
    },
  });

  const { data: ragStats } = useQuery({
    queryKey: ["rag-stats", workspaceId],
    queryFn: () => api.get<RAGStats>(`/rag/stats/${workspaceId}`),
    enabled: hasValidWorkspaceId,
  });

  // -----------------------------------------------------------------------
  // Refresh ragStats when processing finishes
  // -----------------------------------------------------------------------
  const processingCount = useMemo(
    () =>
      documents?.filter((d) => PROCESSING_STATUSES.has(d.status)).length ?? 0,
    [documents]
  );

  const prevProcessingRef = useRef(processingCount);
  useEffect(() => {
    if (prevProcessingRef.current > 0 && processingCount === 0) {
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
    }
    prevProcessingRef.current = processingCount;
  }, [processingCount, queryClient, workspaceId]);

  // Keep selectedDoc in sync with latest document data
  useEffect(() => {
    if (selectedDoc && documents) {
      const updated = documents.find((d) => d.id === selectedDoc.id);
      if (updated && updated.status !== selectedDoc.status) {
        selectDoc(updated);
      }
    }
  }, [documents, selectedDoc, selectDoc]);

  const hasIndexedDocs = (ragStats?.indexed_documents ?? 0) > 0;
  const hasExploreRagDocs = (ragStats?.explorerag_documents ?? 0) > 0;

  // -----------------------------------------------------------------------
  // Mutations
  // -----------------------------------------------------------------------
  const uploadDoc = useMutation({
    mutationFn: ({ file, metadata }: { file: File; metadata: Record<string, unknown> }) =>
      api.uploadFile<Document>(`/documents/upload/${workspaceId}`, file, metadata),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
    onError: () => toast.error("文档上传失败"),
  });

  const updateDocumentMetadata = useMutation({
    mutationFn: ({ document, metadata }: { document: Document; metadata: Record<string, unknown> }) =>
      api.patch<Document>(`/documents/${document.id}/metadata`, {
        custom_metadata: metadata,
        metadata_revision: document.metadata_revision,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      toast.success("元数据已更新");
    },
    onError: (error: Error) => toast.error(error.message || "元数据更新失败"),
  });

  const deleteDoc = useMutation({
    mutationFn: (docId: number) => api.delete(`/documents/${docId}`),
    onSuccess: (_, docId) => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      if (selectedDoc?.id === docId) selectDoc(null);
      toast.success("文档已删除");
    },
    onError: () => toast.error("删除文档失败"),
  });

  const processDoc = useMutation({
    mutationFn: (docId: number) => api.post(`/rag/process/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      toast.info("正在处理文档...", {
        description: "正在解析内容并构建检索索引。",
      });
    },
    onError: (error: Error) => {
      if (error.message?.includes("already being analyzed")) {
        toast.info("文档正在处理中", {
          description: "请等待当前处理完成。",
        });
        // Refresh to get latest status
        queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      } else {
        toast.error("启动文档处理失败");
      }
    },
  });

  const reindexDoc = useMutation({
    mutationFn: (docId: number) => api.post(`/rag/reindex/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      toast.success("已开始重新处理文档");
    },
    onError: (error: Error) => {
      if (error.message?.includes("currently being processed")) {
        toast.info("文档正在处理中，请等待当前任务完成。");
      } else {
        toast.error("重新处理文档失败");
      }
    },
  });

  // -----------------------------------------------------------------------
  // Handlers
  // -----------------------------------------------------------------------
  const handleSelectDoc = useCallback(
    (doc: Document) => {
      if (doc.status !== "indexed") return;
      if (selectedDoc?.id === doc.id) {
        selectDoc(null);
      } else {
        selectDoc(doc);
      }
    },
    [selectedDoc, selectDoc]
  );

  const handleUpdateWorkspace = useCallback(
    async (data: UpdateWorkspace) => {
      if (!wsId) return;
      await updateWorkspace.mutateAsync({ id: wsId, data });
    },
    [wsId, updateWorkspace]
  );

  // A direct deep link starts with no client-side workspace state. Render a
  // deliberate loading/error state instead of an empty three-column shell.
  if (!hasValidWorkspaceId) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm font-medium">知识库地址无效</p>
        <button className="text-sm text-primary hover:underline" onClick={() => navigate("/")}>
          返回知识库
        </button>
      </div>
    );
  }

  if (workspaceLoading) {
    return (
      <div className="h-full grid grid-cols-[minmax(220px,20%)_minmax(300px,40%)_minmax(300px,40%)]">
        <div className="border-r p-4 space-y-4"><div className="h-8 w-3/4 rounded bg-muted" /><div className="h-32 rounded bg-muted" /></div>
        <div className="border-r p-4 space-y-4"><div className="h-8 w-1/3 rounded bg-muted" /><div className="h-5 rounded bg-muted" /><div className="h-5 w-4/5 rounded bg-muted" /></div>
        <div className="p-4 space-y-4"><div className="h-8 w-1/3 rounded bg-muted" /><div className="h-48 rounded bg-muted" /></div>
      </div>
    );
  }

  if (workspaceError || !workspace) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-sm font-medium">无法打开此知识库</p>
        <p className="text-xs text-muted-foreground">该知识库可能已删除，或后端服务不可用。</p>
        <button className="text-sm text-primary hover:underline" onClick={() => navigate("/")}>
          返回知识库
        </button>
      </div>
    );
  }

  // -----------------------------------------------------------------------
  // Render — 3-column layout
  // -----------------------------------------------------------------------
  return (
    <div ref={layoutRef} className="h-full flex overflow-hidden">
      <div className="h-full min-w-0 flex-shrink-0" style={{ width: columnWidths.left }}>
        {/* Column 1: Data Area */}
        <DataPanel
          workspace={workspace}
          documents={documents}
          docsLoading={docsLoading}
          selectedDocId={selectedDoc?.id ?? null}
          onSelectDoc={handleSelectDoc}
          onUpload={async (files, metadata) => {
            await Promise.all(files.map((file) => uploadDoc.mutateAsync({ file, metadata })));
            toast.success(`已上传 ${files.length} 个文档`);
          }}
          isUploading={uploadDoc.isPending || updateDocumentMetadata.isPending}
          onUpdateMetadata={async (document, metadata) => {
            await updateDocumentMetadata.mutateAsync({ document, metadata });
          }}
          onDelete={(id) => deleteDoc.mutate(id)}
          onProcess={(id) => processDoc.mutate(id)}
          onReindex={(id) => reindexDoc.mutate(id)}
          isProcessing={processDoc.isPending}
          onUpdateWorkspace={handleUpdateWorkspace}
        />
      </div>

      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="调整左侧栏宽度"
        onPointerDown={(event) => startResize("left", event)}
        className="group relative w-1.5 flex-shrink-0 cursor-col-resize touch-none bg-border/70 hover:bg-primary/60 active:bg-primary"
      >
        <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border group-hover:bg-primary/70" />
      </div>

      <div
        className={visualPanelOpen ? "h-full min-w-0 flex-shrink-0" : "h-full min-w-0 flex-1"}
        style={visualPanelOpen ? { width: columnWidths.chat } : undefined}
      >
        {/* Column 2: Chat Area */}
        <ChatPanel
          workspaceId={workspaceId || ""}
          hasIndexedDocs={hasIndexedDocs}
          workspace={workspace ?? null}
        />
      </div>

      {visualPanelOpen && (
        <>
          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="调整对话栏宽度"
            onPointerDown={(event) => startResize("chat", event)}
            className="group relative w-1.5 flex-shrink-0 cursor-col-resize touch-none bg-border/70 hover:bg-primary/60 active:bg-primary"
          >
            <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-border group-hover:bg-primary/70" />
          </div>

          <div ref={visualPanelRef} className="h-full min-w-[320px] flex-1">
            {/* Column 3: Visual Area */}
            <VisualPanel
              workspaceId={workspaceId || ""}
              hasExploreRagDocs={hasExploreRagDocs}
            />
          </div>
        </>
      )}
    </div>
  );
}
