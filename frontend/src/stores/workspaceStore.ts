/**
 * Workspace Store
 * ===============
 * Zustand store for the 3-column workspace layout.
 * Manages: document selection, visual panel tabs, citation highlights.
 *
 * Scroll strategy — 2-phase:
 *   Phase 1: open content view (selectDoc + activeTab) — clears scroll targets
 *   Phase 2: after content stabilises, set scroll targets (triggers scroll effect)
 *   When content is already visible for the same doc → skip to Phase 2 (immediate).
 */

import { create } from "zustand";
import type {
  ChatImageRef,
  ChatSourceChunk,
  Document,
} from "@/types";

export type VisualTab = "content" | "kg";
export type KGSubTab = "graph" | "entities";

// Module-level timer so we can cancel stale deferred scrolls
let _scrollTimer: ReturnType<typeof setTimeout> | null = null;

/** How long to wait for content to render before applying scroll targets (ms) */
const SCROLL_DEFER_MS = 350;

interface WorkspaceState {
  // Document selection
  selectedDoc: Document | null;
  visualPanelOpen: boolean;

  // Visual panel tabs
  activeTab: VisualTab;
  kgSubTab: KGSubTab;

  // Scroll targets (for citation → content navigation)
  scrollToPage: number | null;
  scrollToHeading: string | null;
  scrollToImageSrc: string | null;

  // Citation highlights
  highlightChunks: ChatSourceChunk[];
  highlightEntities: string[];
  highlightDocumentIds: number[];
  activeCitationIndex: number | string | null;

  // Actions
  selectDoc: (doc: Document | null) => void;
  setVisualPanelOpen: (open: boolean) => void;
  setActiveTab: (tab: VisualTab) => void;
  setKgSubTab: (tab: KGSubTab) => void;
  activateCitation: (source: ChatSourceChunk, allEntities: string[], doc?: Document) => void;
  activateCitationKG: (source: ChatSourceChunk, allEntities: string[], doc?: Document) => void;
  activateImageCitation: (imageRef: ChatImageRef, doc?: Document) => void;
  clearHighlights: () => void;
  clearScrollTarget: () => void;
  /** Reset all state — call when workspaceId changes */
  reset: () => void;
}

function cancelPendingScroll() {
  if (_scrollTimer) {
    clearTimeout(_scrollTimer);
    _scrollTimer = null;
  }
}

export const useWorkspaceStore = create<WorkspaceState>((set, get) => ({
  selectedDoc: null,
  visualPanelOpen: false,
  activeTab: "content",
  kgSubTab: "graph",
  scrollToPage: null,
  scrollToHeading: null,
  scrollToImageSrc: null,
  highlightChunks: [],
  highlightEntities: [],
  highlightDocumentIds: [],
  activeCitationIndex: null,

  selectDoc: (doc) => {
    cancelPendingScroll();
    set({
      selectedDoc: doc,
      ...(doc ? { visualPanelOpen: true } : {}),
      activeTab: "content",
      highlightChunks: [],
      highlightEntities: [],
      highlightDocumentIds: [],
      activeCitationIndex: null,
      scrollToPage: null,
      scrollToHeading: null,
      scrollToImageSrc: null,
    });
  },

  setActiveTab: (tab) => set({ activeTab: tab }),
  setVisualPanelOpen: (open) => set({ visualPanelOpen: open }),
  setKgSubTab: (tab) => set({ kgSubTab: tab }),

  activateCitation: (source, allEntities, doc) => {
    cancelPendingScroll();

    const state = get();
    const contentAlreadyOpen =
      state.activeTab === "content" &&
      state.selectedDoc != null &&
      (!doc || state.selectedDoc.id === doc.id);

    const scrollTargets = {
      scrollToPage: source.page_no || null,
      scrollToHeading:
        source.heading_path.length > 0
          ? source.heading_path[source.heading_path.length - 1]
          : null,
      scrollToImageSrc: null as string | null,
    };

    // Phase 1: open content + set highlights (always)
    set({
      ...(doc ? { selectedDoc: doc } : {}),
      visualPanelOpen: true,
      highlightChunks: source.source_type === "kg" ? [] : [source],
      highlightEntities: allEntities,
      highlightDocumentIds: source.source_type === "kg"
        ? (source.graph_document_ids ?? [])
        : (source.document_id > 0 ? [source.document_id] : []),
      activeCitationIndex: source.index,
      activeTab: "content",
      // If content already open → scroll immediately; otherwise clear and defer
      ...(contentAlreadyOpen
        ? scrollTargets
        : { scrollToPage: null, scrollToHeading: null, scrollToImageSrc: null }),
    });

    // Phase 2: deferred scroll after content renders
    if (!contentAlreadyOpen) {
      _scrollTimer = setTimeout(() => {
        _scrollTimer = null;
        set(scrollTargets);
      }, SCROLL_DEFER_MS);
    }
  },

  activateCitationKG: (source, allEntities, doc) => {
    cancelPendingScroll();
    set({
      ...(doc ? { selectedDoc: doc } : {}),
      visualPanelOpen: true,
      highlightChunks: source.source_type === "kg" ? [] : [source],
      highlightEntities: allEntities,
      highlightDocumentIds: source.source_type === "kg"
        ? (source.graph_document_ids ?? [])
        : (source.document_id > 0 ? [source.document_id] : []),
      activeCitationIndex: source.index,
      activeTab: "kg",
      kgSubTab: "graph",
    });
  },

  activateImageCitation: (imageRef, doc) => {
    cancelPendingScroll();

    const state = get();
    const contentAlreadyOpen =
      state.activeTab === "content" &&
      state.selectedDoc != null &&
      (!doc || state.selectedDoc.id === doc.id);

    const scrollTargets = {
      scrollToPage: imageRef.page_no || null,
      scrollToImageSrc: imageRef.url,
      scrollToHeading: null as string | null,
    };

    set({
      ...(doc ? { selectedDoc: doc } : {}),
      visualPanelOpen: true,
      activeTab: "content",
      highlightChunks: [],
      highlightEntities: [],
      highlightDocumentIds: [],
      activeCitationIndex: null,
      ...(contentAlreadyOpen
        ? scrollTargets
        : { scrollToPage: null, scrollToHeading: null, scrollToImageSrc: null }),
    });

    if (!contentAlreadyOpen) {
      _scrollTimer = setTimeout(() => {
        _scrollTimer = null;
        set(scrollTargets);
      }, SCROLL_DEFER_MS);
    }
  },

  clearHighlights: () =>
    set({
      highlightChunks: [],
      highlightEntities: [],
      highlightDocumentIds: [],
      activeCitationIndex: null,
    }),

  clearScrollTarget: () =>
    set({ scrollToPage: null, scrollToHeading: null, scrollToImageSrc: null }),

  reset: () => {
    cancelPendingScroll();
    set({
      selectedDoc: null,
      visualPanelOpen: false,
      activeTab: "content",
      kgSubTab: "graph",
      scrollToPage: null,
      scrollToHeading: null,
      scrollToImageSrc: null,
      highlightChunks: [],
      highlightEntities: [],
      highlightDocumentIds: [],
      activeCitationIndex: null,
    });
  },
}));
