-- Cleaned up schema.sql

CREATE TYPE public.documentstatus AS ENUM (
    'PENDING',
    'PARSING',
    'PROCESSING',
    'INDEXING',
    'INDEXED',
    'FAILED'
);

CREATE TABLE public.alembic_version (
    version_num character varying(32) PRIMARY KEY
);

CREATE TABLE public.knowledge_bases (
    id SERIAL PRIMARY KEY,
    name character varying(255) NOT NULL,
    description text,
    system_prompt text,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    kg_language character varying(50),
    kg_entity_types json,
    llm_mode character varying(16) NOT NULL DEFAULT 'cloud'
        CONSTRAINT ck_knowledge_bases_llm_mode CHECK (llm_mode IN ('cloud', 'local')),
    lightrag_augmentation_enabled boolean NOT NULL DEFAULT false,
    chat_cleanup_epoch integer NOT NULL DEFAULT 0,
    metadata_schema jsonb NOT NULL DEFAULT '{"version": 1, "fields": []}'::jsonb,
    metadata_schema_version integer NOT NULL DEFAULT 1
);

CREATE TABLE public.chat_messages (
    id SERIAL PRIMARY KEY,
    workspace_id integer NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    message_id character varying(50) NOT NULL,
    role character varying(20) NOT NULL,
    content text NOT NULL,
    sources json,
    related_entities json,
    image_refs json,
    thinking text,
    agent_steps json,
    reply_to_message_id character varying(50),
    feedback_rating integer,
    feedback_comment text,
    source_ratings json,
    feedback_corrected_answer text,
    feedback_reference_chunk_ids json,
    feedback_failure_types json,
    feedback_review_status character varying(20),
    feedback_promoted_case_id integer,
    created_at timestamp without time zone NOT NULL
);

CREATE INDEX ix_chat_messages_id ON public.chat_messages USING btree (id);
CREATE INDEX ix_chat_messages_message_id ON public.chat_messages USING btree (message_id);
CREATE INDEX ix_chat_messages_workspace_id ON public.chat_messages USING btree (workspace_id);
CREATE INDEX ix_chat_messages_reply_to_message_id ON public.chat_messages USING btree (reply_to_message_id);

CREATE TABLE public.eval_cases (
    id SERIAL PRIMARY KEY,
    workspace_id integer NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    status character varying(20) NOT NULL DEFAULT 'draft',
    source character varying(20) NOT NULL DEFAULT 'manual',
    dataset_name character varying(100) NOT NULL DEFAULT 'core',
    dataset_version integer NOT NULL DEFAULT 1,
    split character varying(20) NOT NULL DEFAULT 'dev',
    is_frozen boolean NOT NULL DEFAULT false,
    category character varying(50) NOT NULL DEFAULT 'other',
    difficulty character varying(20) NOT NULL DEFAULT 'medium',
    expected_behavior character varying(20) NOT NULL DEFAULT 'answer',
    review_status character varying(20) NOT NULL DEFAULT 'draft',
    reviewed_by character varying(100),
    reviewed_at timestamp without time zone,
    question text NOT NULL,
    reference_answer text,
    reference_chunk_ids json,
    reference_contexts json,
    reference_entity_names json,
    reference_relationships json,
    conversation_history json,
    tags json,
    metadata json,
    input_hash character varying(64) NOT NULL,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);
CREATE INDEX ix_eval_cases_workspace_id ON public.eval_cases USING btree (workspace_id);
CREATE INDEX ix_eval_cases_status ON public.eval_cases USING btree (status);
CREATE INDEX ix_eval_cases_source ON public.eval_cases USING btree (source);
CREATE INDEX ix_eval_cases_input_hash ON public.eval_cases USING btree (input_hash);
CREATE INDEX ix_eval_cases_dataset_name ON public.eval_cases USING btree (dataset_name);
CREATE INDEX ix_eval_cases_split ON public.eval_cases USING btree (split);
CREATE INDEX ix_eval_cases_category ON public.eval_cases USING btree (category);
CREATE INDEX ix_eval_cases_review_status ON public.eval_cases USING btree (review_status);

CREATE TABLE public.eval_runs (
    id SERIAL PRIMARY KEY,
    workspace_id integer NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    status character varying(20) NOT NULL DEFAULT 'queued',
    run_type character varying(20) NOT NULL DEFAULT 'fast',
    name character varying(200),
    experiment_id character varying(64),
    variant character varying(20) NOT NULL DEFAULT 'custom',
    dataset_name character varying(100),
    dataset_version integer,
    dataset_split character varying(20),
    case_ids json NOT NULL,
    config json,
    target_config json,
    metrics_summary json,
    error_message text,
    baseline_run_id integer REFERENCES public.eval_runs(id) ON DELETE SET NULL,
    is_baseline boolean NOT NULL DEFAULT false,
    started_at timestamp without time zone,
    finished_at timestamp without time zone,
    created_at timestamp without time zone NOT NULL
);
CREATE INDEX ix_eval_runs_workspace_id ON public.eval_runs USING btree (workspace_id);
CREATE INDEX ix_eval_runs_status ON public.eval_runs USING btree (status);
CREATE INDEX ix_eval_runs_experiment_id ON public.eval_runs USING btree (experiment_id);
CREATE INDEX ix_eval_runs_variant ON public.eval_runs USING btree (variant);
CREATE INDEX ix_eval_runs_dataset_name ON public.eval_runs USING btree (dataset_name);

CREATE TABLE public.eval_results (
    id SERIAL PRIMARY KEY,
    run_id integer NOT NULL REFERENCES public.eval_runs(id) ON DELETE CASCADE,
    case_id integer NOT NULL REFERENCES public.eval_cases(id) ON DELETE CASCADE,
    question text NOT NULL,
    reference_answer text,
    reference_chunk_ids json,
    retrieved_contexts json,
    answer text,
    sources json,
    performance json,
    retrieval_trace json,
    metrics json,
    metric_status json,
    metric_details json,
    failure_types json,
    baseline_delta json,
    review_status character varying(20) NOT NULL DEFAULT 'unreviewed',
    reviewer_verdict character varying(20),
    reviewer_comment text,
    verdict character varying(20) NOT NULL DEFAULT 'pending',
    error_message text,
    created_at timestamp without time zone NOT NULL
);
CREATE INDEX ix_eval_results_run_id ON public.eval_results USING btree (run_id);
CREATE INDEX ix_eval_results_case_id ON public.eval_results USING btree (case_id);

CREATE TYPE public.chatattachmentstate AS ENUM (
    'UPLOADED', 'QUEUED', 'PARSING', 'READY_DIRECT', 'INDEXED_TEMP',
    'FAILED', 'CLEARING', 'DELETED'
);

CREATE TABLE public.chat_attachments (
    id character varying(36) PRIMARY KEY,
    workspace_id integer NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    original_filename character varying(255) NOT NULL,
    file_type character varying(20) NOT NULL,
    file_size integer NOT NULL,
    storage_path character varying(1000) NOT NULL,
    artifact_dir character varying(1000) NOT NULL,
    state public.chatattachmentstate NOT NULL,
    parsed_token_count integer NOT NULL DEFAULT 0,
    temp_collection character varying(255),
    error_message text,
    cleanup_epoch integer NOT NULL DEFAULT 0,
    cleanup_pending boolean NOT NULL DEFAULT false,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);

CREATE INDEX ix_chat_attachments_workspace_id ON public.chat_attachments USING btree (workspace_id);
CREATE INDEX ix_chat_attachments_state ON public.chat_attachments USING btree (state);

CREATE TABLE public.chat_message_attachments (
    id SERIAL PRIMARY KEY,
    message_id integer NOT NULL REFERENCES public.chat_messages(id) ON DELETE CASCADE,
    attachment_id character varying(36) NOT NULL REFERENCES public.chat_attachments(id) ON DELETE CASCADE,
    created_at timestamp without time zone NOT NULL
);

CREATE INDEX ix_chat_message_attachments_message_id ON public.chat_message_attachments USING btree (message_id);
CREATE INDEX ix_chat_message_attachments_attachment_id ON public.chat_message_attachments USING btree (attachment_id);

CREATE TABLE public.documents (
    id SERIAL PRIMARY KEY,
    workspace_id integer NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    filename character varying(255) NOT NULL,
    original_filename character varying(255) NOT NULL,
    file_type character varying(50) NOT NULL,
    file_size integer NOT NULL,
    status public.documentstatus NOT NULL,
    chunk_count integer NOT NULL,
    error_message character varying(500),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    markdown_content text,
    page_count integer NOT NULL,
    image_count integer NOT NULL,
    table_count integer NOT NULL,
    parser_version character varying(50),
    processing_time_ms integer NOT NULL,
    custom_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    processing_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata_revision integer NOT NULL DEFAULT 1,
    content_version integer NOT NULL DEFAULT 1,
    content_sha256 character varying(64),
    metadata_requires_reindex boolean NOT NULL DEFAULT false,
    kg_document_id character varying(128) UNIQUE,
    kg_index_status character varying(20) NOT NULL DEFAULT 'not_indexed',
    kg_indexed_content_version integer NOT NULL DEFAULT 0
);

CREATE INDEX ix_documents_custom_metadata_gin ON public.documents USING gin (custom_metadata);

CREATE TABLE public.document_images (
    id SERIAL PRIMARY KEY,
    document_id integer NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    image_id character varying(100) NOT NULL UNIQUE,
    page_no integer NOT NULL,
    file_path character varying(500) NOT NULL,
    caption text NOT NULL,
    width integer NOT NULL,
    height integer NOT NULL,
    mime_type character varying(50) NOT NULL,
    created_at timestamp without time zone NOT NULL
);

CREATE TABLE public.document_tables (
    id SERIAL PRIMARY KEY,
    document_id integer NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    table_id character varying(100) NOT NULL UNIQUE,
    page_no integer NOT NULL,
    content_markdown text NOT NULL,
    caption text NOT NULL,
    num_rows integer NOT NULL,
    num_cols integer NOT NULL,
    created_at timestamp without time zone NOT NULL
);
