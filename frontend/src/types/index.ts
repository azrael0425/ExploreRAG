// Knowledge Base (Document Workspace)
export type LLMMode = "cloud" | "local";

export type MetadataFieldType =
  | "string"
  | "integer"
  | "number"
  | "boolean"
  | "date"
  | "datetime"
  | "enum"
  | "multi_enum";

export interface MetadataFieldDefinition {
  key: string;
  label: string;
  type: MetadataFieldType;
  required: boolean;
  filterable: boolean;
  semantic: boolean;
  options: string[];
  default?: unknown;
}

export interface MetadataSchema {
  version: number;
  fields: MetadataFieldDefinition[];
}

export interface MetadataFilterRule {
  field: string;
  op: "eq" | "neq" | "exists" | "gt" | "gte" | "lt" | "lte" | "between" | "in" | "not_in" | "contains_any" | "contains_all";
  value?: unknown;
}

export interface MetadataFilter {
  and?: MetadataFilterRule[];
  or?: MetadataFilterRule[];
}

export interface KnowledgeBase {
  id: number;
  name: string;
  description: string | null;
  system_prompt: string | null;
  kg_language: string | null;
  kg_entity_types: string[] | null;
  llm_mode: LLMMode;
  lightrag_augmentation_enabled: boolean;
  lightrag_available: boolean;
  metadata_schema: MetadataSchema;
  metadata_schema_version: number;
  document_count: number;
  indexed_count: number;
  created_at: string;
  updated_at: string;
}

export interface CreateWorkspace {
  name: string;
  description?: string;
}

export interface UpdateWorkspace {
  name?: string;
  description?: string;
  system_prompt?: string | null;
  kg_language?: string | null;
  kg_entity_types?: string[] | null;
  llm_mode?: LLMMode;
  lightrag_augmentation_enabled?: boolean;
}

export interface WorkspaceSummary {
  id: number;
  name: string;
  document_count: number;
  llm_mode: LLMMode;
}

export interface LLMRuntimeStatus {
  mode: LLMMode;
  provider: string;
  model: string;
  available: boolean;
  detail: string | null;
}

export interface Document {
  id: number;
  workspace_id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  status: DocumentStatus;
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  page_count?: number;
  image_count?: number;
  table_count?: number;
  parser_version?: string;
  processing_time_ms?: number;
  custom_metadata: Record<string, unknown>;
  processing_metadata?: Record<string, unknown>;
  metadata_revision: number;
  content_version?: number;
  kg_document_id?: string | null;
  kg_index_status?: string;
  metadata_requires_reindex?: boolean;
}

// RAG Types
export type DocumentStatus = "pending" | "parsing" | "indexing" | "processing" | "indexed" | "failed";

export type RAGQueryMode = "hybrid" | "vector_only" | "local" | "global";

export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  document_ids?: number[];
  metadata_filter?: MetadataFilter;
  mode?: RAGQueryMode;
}

export interface Citation {
  source_file: string;
  document_id: number | null;
  page_no: number | null;
  heading_path: string[];
  formatted: string;
}

export interface DocumentImage {
  image_id: string;
  document_id: number;
  page_no: number;
  caption: string;
  width: number;
  height: number;
  url: string;
}

export interface RetrievedChunk {
  content: string;
  chunk_id: string;
  score: number;
  metadata: Record<string, unknown>;
  citation?: Citation;
}

export interface RAGQueryResponse {
  query: string;
  chunks: RetrievedChunk[];
  context: string;
  total_chunks: number;
  knowledge_graph_summary?: string;
  citations?: Citation[];
  image_refs?: DocumentImage[];
}

export interface RAGStats {
  workspace_id: number;
  total_documents: number;
  indexed_documents: number;
  total_chunks: number;
  image_count?: number;
  explorerag_documents?: number;
}

// Knowledge Graph Types
export interface KGEntity {
  name: string;
  entity_type: string;
  description: string;
  degree: number;
}

export interface KGRelationship {
  source: string;
  target: string;
  description: string;
  keywords: string;
  weight: number;
}

export interface KGGraphNode {
  id: string;
  label: string;
  entity_type: string;
  description: string;
  degree: number;
  source_document_ids?: number[];
  source_files?: string[];
}

export interface KGGraphEdge {
  source: string;
  target: string;
  label: string;
  weight: number;
  source_document_ids?: number[];
  source_files?: string[];
}

export interface KGGraphData {
  nodes: KGGraphNode[];
  edges: KGGraphEdge[];
  is_truncated: boolean;
}

export interface KGFocusGraphData extends KGGraphData {
  requested_entities: string[];
  matched_entities: string[];
  missing_entities: string[];
  selection_mode: "citation_focus";
}

// Chat Types
export interface ChatImageRef {
  ref_id?: string;  // 4-char alphanumeric ID, e.g. "p4f2"
  image_id: string;
  document_id: number;
  attachment_id?: string | null;
  page_no: number;
  caption: string;
  url: string;
  width: number;
  height: number;
}

export interface RAGPerformanceMetrics {
  vector_ms?: number | null;
  graph_ms?: number | null;
  rerank_ms?: number | null;
  context_ms?: number | null;
  generation_ms?: number | null;
  first_token_ms?: number | null;
  postprocess_ms?: number | null;
  total_ms: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[];
  relatedEntities?: string[];
  imageRefs?: ChatImageRef[];
  thinking?: string | null;
  timestamp: string;
  isStreaming?: boolean;
  agentSteps?: AgentStep[];
  performance?: RAGPerformanceMetrics;
  attachments?: ChatAttachment[];
  feedbackRating?: number | null;
  sourceRatings?: Record<string, number>;
}

export type ChatAttachmentState =
  | "uploaded"
  | "queued"
  | "parsing"
  | "ready_direct"
  | "indexed_temp"
  | "failed"
  | "clearing"
  | "deleted";

export interface ChatAttachment {
  attachment_id: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  state: ChatAttachmentState;
  parsed_token_count: number;
  error_message?: string | null;
  created_at: string;
}

export interface ChatSourceChunk {
  index: number | string;  // number for legacy, string for new [a3x9] format
  chunk_id: string;
  content: string;
  document_id: number;
  attachment_id?: string | null;
  source_file?: string;
  page_no: number;
  heading_path: string[];
  score: number;
  source_type?: "vector" | "kg" | "attachment";
  graph_entity_names?: string[];
  graph_document_ids?: number[];
}

export interface ChatResponseData {
  answer: string;
  sources: ChatSourceChunk[];
  related_entities: string[];
  kg_summary: string | null;
  image_refs: ChatImageRef[];
  thinking: string | null;
  performance?: RAGPerformanceMetrics;
}

export interface PersistedChatMessage {
  id: number;
  message_id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[] | null;
  related_entities?: string[] | null;
  image_refs?: ChatImageRef[] | null;
  thinking?: string | null;
  agent_steps?: AgentStep[] | null;
  reply_to_message_id?: string | null;
  feedback_rating?: number | null;
  feedback_comment?: string | null;
  source_ratings?: Record<string, number> | null;
  attachments?: ChatAttachment[] | null;
  created_at: string;
}

export interface ChatHistoryResponse {
  workspace_id: number;
  messages: PersistedChatMessage[];
  total: number;
}

export interface LLMCapabilities {
  provider: string;
  model: string;
  supports_thinking: boolean;
  supports_vision: boolean;
  thinking_default: boolean;
}

// SSE Streaming Types
export type ChatStreamStatus = "idle" | "analyzing" | "retrieving" | "generating" | "error";

// Agent Step Types (ThinkingTimeline)
export type AgentStepType =
  | "analyzing"
  | "understood"
  | "retrieving"
  | "sources_found"
  | "generating"
  | "attachment"
  | "done"
  | "error";

export type AgentStepStatus = "active" | "completed" | "error";

export interface AgentStep {
  id: string;
  step: AgentStepType;
  detail: string;
  status: AgentStepStatus;
  timestamp: number;
  durationMs?: number;
  thinkingText?: string;
  sourceCount?: number;
  imageCount?: number;
  performance?: RAGPerformanceMetrics;
}
