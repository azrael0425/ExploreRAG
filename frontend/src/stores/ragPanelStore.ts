/**
 * RAG Panel Store
 * ===============
 *
 * Zustand store for managing the document detail panel state.
 * Controls which panel is visible (viewer / gallery / citation),
 * the selected document, and panel width.
 */

import { create } from "zustand";
import type { Document, Citation } from "@/types";

export type PanelView = "viewer" | "gallery" | null;

interface RagPanelState {
  /** Which panel view is active */
  activePanel: PanelView;

  /** The document currently shown in the panel */
  selectedDoc: Document | null;

  /** Optional: scroll-to target when opening from citation */
  scrollToPage: number | null;
  scrollToHeading: string | null;

  /** Open the panel with a specific view and document */
  openPanel: (view: PanelView, doc: Document) => void;

  /** Switch view without changing document */
  setView: (view: PanelView) => void;

  /** Close the panel */
  closePanel: () => void;

  /** Open viewer at a specific citation location */
  openAtCitation: (doc: Document, citation: Citation) => void;

  /** Clear scroll targets after navigation */
  clearScrollTarget: () => void;
}

export const useRagPanelStore = create<RagPanelState>((set) => ({
  activePanel: null,
  selectedDoc: null,
  scrollToPage: null,
  scrollToHeading: null,

  openPanel: (view, doc) =>
    set({
      activePanel: view,
      selectedDoc: doc,
      scrollToPage: null,
      scrollToHeading: null,
    }),

  setView: (view) => set({ activePanel: view }),

  closePanel: () =>
    set({
      activePanel: null,
      selectedDoc: null,
      scrollToPage: null,
      scrollToHeading: null,
    }),

  openAtCitation: (doc, citation) =>
    set({
      activePanel: "viewer",
      selectedDoc: doc,
      scrollToPage: citation.page_no,
      scrollToHeading: citation.heading_path.length > 0
        ? citation.heading_path[citation.heading_path.length - 1]
        : null,
    }),

  clearScrollTarget: () =>
    set({ scrollToPage: null, scrollToHeading: null }),
}));
