import { memo } from "react";
import {
  BookOpen,
  Network,
  List,
  FileSearch,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { DocumentViewer } from "./DocumentViewer";
import { KnowledgeGraphView } from "./KnowledgeGraphView";
import { EntityList } from "./EntityList";

// ---------------------------------------------------------------------------
// Tab button
// ---------------------------------------------------------------------------
function TabButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
        active
          ? "bg-primary/15 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Sub-tab button (smaller, for KG inner tabs)
// ---------------------------------------------------------------------------
function SubTabButton({
  active,
  icon,
  label,
  onClick,
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded transition-colors",
        active
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------
function EmptyVisual() {
  return (
    <div className="h-full flex flex-col items-center justify-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
        <FileSearch className="w-7 h-7 text-muted-foreground/40" />
      </div>
      <p className="text-sm font-medium text-muted-foreground">
        请选择要查看的文档
      </p>
      <p className="text-xs text-muted-foreground/60 mt-1 text-center max-w-[200px]">
        点击左侧就绪的文档，即可查看其内容
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// KG Content — graph or entities
// ---------------------------------------------------------------------------
const KGContent = memo(function KGContent({
  workspaceId,
  highlightEntities,
  highlightDocumentIds,
  citationId,
  onClearFocus,
}: {
  workspaceId: string;
  highlightEntities: string[];
  highlightDocumentIds: number[];
  citationId: number | string | null;
  onClearFocus: () => void;
}) {
  const { kgSubTab, setKgSubTab } = useWorkspaceStore();

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Sub-tab bar */}
      <div className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 border-b bg-muted/20">
        <SubTabButton
          active={kgSubTab === "graph"}
          icon={<Network className="w-3 h-3" />}
          label="图谱"
          onClick={() => setKgSubTab("graph")}
        />
        <SubTabButton
          active={kgSubTab === "entities"}
          icon={<List className="w-3 h-3" />}
          label="实体"
          onClick={() => setKgSubTab("entities")}
        />
      </div>

      {/* Content */}
      {kgSubTab === "graph" ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <KnowledgeGraphView
            projectId={workspaceId}
            highlightEntities={highlightEntities}
            highlightDocumentIds={highlightDocumentIds}
            citationId={citationId}
            onClearFocus={onClearFocus}
          />
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-hidden p-3">
          <EntityList
            projectId={workspaceId}
            highlightEntities={highlightEntities}
          />
        </div>
      )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// VisualPanel — main export
// ---------------------------------------------------------------------------
interface VisualPanelProps {
  workspaceId: string;
  hasExploreRagDocs: boolean;
}

export const VisualPanel = memo(function VisualPanel({
  workspaceId,
  hasExploreRagDocs,
}: VisualPanelProps) {
  const {
    selectedDoc,
    activeTab,
    setActiveTab,
    scrollToPage,
    scrollToHeading,
    scrollToImageSrc,
    highlightChunks,
    highlightEntities,
    highlightDocumentIds,
    activeCitationIndex,
    clearHighlights,
    clearScrollTarget,
  } = useWorkspaceStore();

  if (!selectedDoc && activeTab !== "kg") return <EmptyVisual />;

  return (
    <div className="h-full flex flex-col overflow-hidden min-h-0">
      {/* Tab bar */}
      <div className="flex-shrink-0 flex items-center gap-1 px-3 py-2 border-b">
        <TabButton
          active={activeTab === "content"}
          icon={<BookOpen className="w-3.5 h-3.5" />}
          label="文档内容"
          onClick={() => setActiveTab("content")}
        />
        {hasExploreRagDocs && (
          <TabButton
            active={activeTab === "kg"}
            icon={<Network className="w-3.5 h-3.5" />}
            label="知识图谱"
            onClick={() => setActiveTab("kg")}
          />
        )}
        {/* Active highlights indicator */}
        {activeTab === "content" && highlightChunks.length > 0 && (
          <span className="ml-auto text-[10px] text-primary bg-primary/10 px-2 py-0.5 rounded-full">
             已定位引用片段
          </span>
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "content" && selectedDoc ? (
          <DocumentViewer
            doc={selectedDoc}
            scrollToPage={scrollToPage}
            scrollToHeading={scrollToHeading}
            scrollToImageSrc={scrollToImageSrc}
            highlightChunks={highlightChunks}
            onScrolled={clearScrollTarget}
          />
        ) : (
          <KGContent
            workspaceId={workspaceId}
            highlightEntities={highlightEntities}
            highlightDocumentIds={highlightDocumentIds}
            citationId={activeCitationIndex}
            onClearFocus={clearHighlights}
          />
        )}
      </div>
    </div>
  );
});
