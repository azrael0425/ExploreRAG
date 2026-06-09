import { beforeEach, describe, expect, it } from "vitest";

import type { Citation, Document } from "@/types";
import { useRagPanelStore } from "./ragPanelStore";

const document = {
  id: 7,
  workspace_id: 3,
  filename: "report.pdf",
  original_filename: "report.pdf",
  file_type: "application/pdf",
  file_size: 1024,
  status: "indexed",
  chunk_count: 12,
  error_message: null,
  created_at: "2026-06-09T00:00:00Z",
  updated_at: "2026-06-09T00:00:00Z",
  custom_metadata: {},
  metadata_revision: 1,
} satisfies Document;

beforeEach(() => {
  useRagPanelStore.setState({
    activePanel: null,
    selectedDoc: null,
    scrollToPage: null,
    scrollToHeading: null,
  });
});

describe("RAG panel navigation", () => {
  it("opens a citation at its page and deepest heading", () => {
    const citation: Citation = {
      source_file: "report.pdf",
      document_id: document.id,
      page_no: 9,
      heading_path: ["Security", "Prompt injection"],
      formatted: "report.pdf · p.9",
    };

    useRagPanelStore.getState().openAtCitation(document, citation);

    expect(useRagPanelStore.getState()).toMatchObject({
      activePanel: "viewer",
      selectedDoc: document,
      scrollToPage: 9,
      scrollToHeading: "Prompt injection",
    });
  });

  it("clears all document state when the panel closes", () => {
    useRagPanelStore.getState().openPanel("gallery", document);
    useRagPanelStore.getState().closePanel();

    expect(useRagPanelStore.getState()).toMatchObject({
      activePanel: null,
      selectedDoc: null,
      scrollToPage: null,
      scrollToHeading: null,
    });
  });
});
