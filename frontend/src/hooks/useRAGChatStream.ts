/**
 * useRAGChatStream — SSE streaming hook for ExploreRAG chat.
 *
 * Handles Server-Sent Events from the /chat/{workspace_id}/stream endpoint,
 * with rAF-buffered token rendering, AgentStep tracking, and AbortController cleanup.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { generateId } from "@/lib/utils";
import type {
  ChatSourceChunk,
  ChatImageRef,
  ChatStreamStatus,
  ChatMessage,
  AgentStep,
  AgentStepType,
  RAGPerformanceMetrics,
} from "@/types";
import { BASE_URL } from "@/lib/api";

export interface RAGStreamResult {
  /** Current stream status */
  status: ChatStreamStatus;
  /** Accumulated streaming content (answer text so far) */
  streamingContent: string;
  /** Accumulated thinking text */
  thinkingText: string;
  /** Sources received from retrieval */
  pendingSources: ChatSourceChunk[];
  /** Image refs received from retrieval */
  pendingImages: ChatImageRef[];
  /** Error message if any */
  error: string | null;
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Agent processing steps for ThinkingTimeline */
  agentSteps: AgentStep[];
  /** Send a message — returns the finalized ChatMessage on complete */
  sendMessage: (
    message: string,
    history: { role: string; content: string }[],
    enableThinking: boolean,
    attachmentIds?: string[],
  ) => Promise<ChatMessage | null>;
  /** Cancel ongoing stream */
  cancel: () => void;
  /** Reset all state */
  reset: () => void;
}

// ---------------------------------------------------------------------------
// AgentStep helpers
// ---------------------------------------------------------------------------

function createStep(
  step: AgentStepType,
  detail: string,
  status: "active" | "completed" | "error" = "active",
): AgentStep {
  return {
    id: generateId(),
    step,
    detail,
    status,
    timestamp: Date.now(),
  };
}

function completeActiveStep(steps: AgentStep[]): AgentStep[] {
  const now = Date.now();
  return steps.map((s) =>
    s.status === "active"
      ? { ...s, status: "completed" as const, durationMs: now - s.timestamp }
      : s,
  );
}

function markActiveError(steps: AgentStep[]): AgentStep[] {
  return steps.map((s) =>
    s.status === "active" ? { ...s, status: "error" as const } : s,
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRAGChatStream(workspaceId: string): RAGStreamResult {
  const [status, setStatus] = useState<ChatStreamStatus>("idle");
  const [streamingContent, setStreamingContent] = useState("");
  const [thinkingText, setThinkingText] = useState("");
  const [pendingSources, setPendingSources] = useState<ChatSourceChunk[]>([]);
  const [pendingImages, setPendingImages] = useState<ChatImageRef[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const bufferRef = useRef("");
  const rafRef = useRef<number | undefined>(undefined);

  // Separate thinking text buffer for AgentStep thinkingText updates
  const thinkingBufferRef = useRef("");
  const thinkingRafRef = useRef<number | undefined>(undefined);

  // Track start time for total duration
  const streamStartRef = useRef(0);

  // Cleanup on unmount — abort fetch, cancel reader, and clear pending RAFs
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      readerRef.current?.cancel().catch(() => {});
      readerRef.current = null;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (thinkingRafRef.current) cancelAnimationFrame(thinkingRafRef.current);
    };
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setStreamingContent("");
    setThinkingText("");
    setPendingSources([]);
    setPendingImages([]);
    setError(null);
    setIsStreaming(false);
    setAgentSteps([]);
    bufferRef.current = "";
    thinkingBufferRef.current = "";
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    if (thinkingRafRef.current) {
      cancelAnimationFrame(thinkingRafRef.current);
      thinkingRafRef.current = undefined;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    readerRef.current?.cancel().catch(() => {});
    readerRef.current = null;
    setStatus("idle");
    setIsStreaming(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    if (thinkingRafRef.current) {
      cancelAnimationFrame(thinkingRafRef.current);
      thinkingRafRef.current = undefined;
    }
    // Flush any remaining token buffer
    if (bufferRef.current) {
      const remaining = bufferRef.current;
      bufferRef.current = "";
      setStreamingContent((prev) => prev + remaining);
    }
  }, []);

  const onToken = useCallback((text: string) => {
    bufferRef.current += text;
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(() => {
        const chunk = bufferRef.current;
        bufferRef.current = "";
        rafRef.current = undefined;
        setStreamingContent((prev) => prev + chunk);
      });
    }
  }, []);

  // Buffered thinking text update for the analyzing AgentStep
  const onThinkingToken = useCallback((text: string) => {
    // Update flat thinkingText state (existing behavior)
    setThinkingText((prev) => prev + text);

    // Buffer thinking text for AgentStep update
    thinkingBufferRef.current += text;
    if (!thinkingRafRef.current) {
      thinkingRafRef.current = requestAnimationFrame(() => {
        const chunk = thinkingBufferRef.current;
        thinkingBufferRef.current = "";
        thinkingRafRef.current = undefined;

        setAgentSteps((prev) => {
          // Find the analyzing step regardless of status — thinking can
          // arrive during both the first iteration (analyzing=active) and
          // the second iteration after tool call (analyzing=completed).
          const idx = prev.findIndex((s) => s.step === "analyzing");
          if (idx === -1) return prev;
          const updated = [...prev];
          updated[idx] = {
            ...updated[idx],
            thinkingText: (updated[idx].thinkingText || "") + chunk,
          };
          return updated;
        });
      });
    }
  }, []);

  const sendMessage = useCallback(
    async (
      message: string,
      history: { role: string; content: string }[],
      enableThinking: boolean,
      attachmentIds: string[] = [],
    ): Promise<ChatMessage | null> => {
      // Abort any in-flight request before starting a new one
      abortRef.current?.abort();
      readerRef.current?.cancel().catch(() => {});
      readerRef.current = null;

      // Cancel pending RAFs from previous stream
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = undefined;
      }
      if (thinkingRafRef.current) {
        cancelAnimationFrame(thinkingRafRef.current);
        thinkingRafRef.current = undefined;
      }

      // Reset state for new message
      setStreamingContent("");
      setThinkingText("");
      setPendingSources([]);
      setPendingImages([]);
      setError(null);
      setStatus("analyzing");
      setIsStreaming(true);
      setAgentSteps([]);
      bufferRef.current = "";
      thinkingBufferRef.current = "";
      streamStartRef.current = Date.now();

      // Synchronous local tracker — avoids React 18 batching race condition
      // where agentStepsRef in ChatPanel may be stale when sendMessage resolves
      let localSteps: AgentStep[] = [];
      // Accumulate all thinking text in this scope so it can be flushed into
      // localSteps at complete time (onThinkingToken only updates setAgentSteps
      // via RAF, which never syncs back to localSteps)
      let thinkingAccumulator = "";
      function syncUpdateSteps(updater: AgentStep[] | ((prev: AgentStep[]) => AgentStep[])): void {
        const next = typeof updater === "function" ? updater(localSteps) : updater;
        localSteps = next;
        setAgentSteps(next);
      }

      abortRef.current = new AbortController();

      try {
        const response = await fetch(
          `${BASE_URL}/rag/chat/${workspaceId}/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message,
              history,
              enable_thinking: enableThinking,
              attachment_ids: attachmentIds,
            }),
            signal: abortRef.current.signal,
          },
        );

        if (!response.ok) {
          const err = await response
            .json()
            .catch(() => ({ detail: "流式请求失败" }));
          throw new Error(err.detail || `Error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");
        readerRef.current = reader;

        const decoder = new TextDecoder();
        let sseBuffer = "";
        let currentEventType = "unknown";
        let finalMessage: ChatMessage | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split("\n");
          sseBuffer = lines.pop() || "";

          for (const line of lines) {
            // Skip heartbeat comments
            if (line.startsWith(":")) continue;

            if (line.startsWith("event: ")) {
              currentEventType = line.slice(7).trim();
              continue;
            }

            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;

              try {
                const data = JSON.parse(jsonStr);

                switch (currentEventType) {
                  case "status": {
                    const step = data.step as string;
                    const detail = (data.detail as string) || "";

                    if (step === "analyzing") {
                      setStatus("analyzing");
                      syncUpdateSteps((prev) => [
                        ...prev,
                        createStep("analyzing", detail || "正在分析你的问题…"),
                      ]);
                    } else if (step === "retrieving") {
                      setStatus("retrieving");
                      syncUpdateSteps((prev) => [
                        ...completeActiveStep(prev),
                        createStep("understood", "已理解问题", "completed"),
                        createStep("retrieving", detail || "正在检索知识库…"),
                      ]);
                    } else if (step === "generating") {
                      setStatus("generating");
                      syncUpdateSteps((prev) => [
                        ...completeActiveStep(prev),
                        createStep("generating", detail || "正在生成回答…"),
                      ]);
                    }
                    break;
                  }

                  case "attachment_validating":
                  case "attachment_queued":
                  case "attachment_parsing":
                  case "attachment_ocr_retry":
                  case "attachment_indexing":
                  case "attachment_ready":
                  case "retrieving_attachments": {
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      createStep("attachment", data.detail || "正在处理临时附件", currentEventType === "attachment_ready" ? "completed" : "active"),
                    ]);
                    break;
                  }

                  case "thinking":
                    onThinkingToken(data.text || "");
                    thinkingAccumulator += data.text || "";
                    break;

                  case "sources": {
                    const sources = (data.sources || []) as ChatSourceChunk[];
                    setPendingSources((prev) => [...prev, ...sources]);

                    // Keep source identifiers in citation data only; they are not user-facing.
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      createStep("sources_found", `已找到 ${sources.length} 个来源`, "completed"),
                    ].map((s) =>
                      s.step === "sources_found" && s.status === "completed" && !s.sourceCount
                        ? { ...s, sourceCount: sources.length }
                        : s,
                    ));
                    break;
                  }

                  case "images": {
                    const imgs = (data.image_refs || []) as ChatImageRef[];
                    setPendingImages((prev) => [...prev, ...imgs]);

                    // Update sources_found step with image count
                    if (imgs.length > 0) {
                      syncUpdateSteps((prev) => {
                        let lastSourcesIdx = -1;
                        for (let i = prev.length - 1; i >= 0; i--) {
                          if (prev[i].step === "sources_found") {
                            lastSourcesIdx = i;
                            break;
                          }
                        }
                        if (lastSourcesIdx === -1) return prev;
                        const updated = [...prev];
                        const existing = updated[lastSourcesIdx];
                        updated[lastSourcesIdx] = {
                          ...existing,
                          imageCount: (existing.imageCount || 0) + imgs.length,
                          detail: `已找到 ${existing.sourceCount || 0} 个来源，关联 ${(existing.imageCount || 0) + imgs.length} 张图片`,
                        };
                        return updated;
                      });
                    }
                    break;
                  }

                  case "token":
                    onToken(data.text || "");
                    break;

                  case "token_rollback":
                    // Clear speculative tokens
                    bufferRef.current = "";
                    if (rafRef.current) {
                      cancelAnimationFrame(rafRef.current);
                      rafRef.current = undefined;
                    }
                    setStreamingContent("");
                    break;

                  case "complete": {
                    // Flush remaining buffer
                    if (bufferRef.current) {
                      bufferRef.current = "";
                      if (rafRef.current) {
                        cancelAnimationFrame(rafRef.current);
                        rafRef.current = undefined;
                      }
                    }
                    // Flush accumulated thinking into localSteps so finalMessage.agentSteps has thinkingText
                    if (thinkingAccumulator) {
                      syncUpdateSteps((prev) =>
                        prev.map((s) =>
                          s.step === "analyzing"
                            ? { ...s, thinkingText: (s.thinkingText || "") + thinkingAccumulator }
                            : s,
                        ),
                      );
                      thinkingAccumulator = "";
                    }
                    // Flush thinking buffer (cancel pending RAF)
                    if (thinkingBufferRef.current) {
                      thinkingBufferRef.current = "";
                      if (thinkingRafRef.current) {
                        cancelAnimationFrame(thinkingRafRef.current);
                        thinkingRafRef.current = undefined;
                      }
                    }

                    // Complete active step + add done step (sync localSteps too)
                    const totalMs = Date.now() - streamStartRef.current;
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      createStep("done", `已完成，用时 ${totalMs >= 1000 ? `${(totalMs / 1000).toFixed(1)}s` : `${totalMs}ms`}`, "completed"),
                    ]);

                    finalMessage = {
                      // The backend persists before emitting `complete`, so
                      // a just-finished answer can accept durable feedback.
                      id: data.message_id || generateId(),
                      role: "assistant",
                      content: data.answer || "",
                      sources: data.sources || [],
                      relatedEntities: data.related_entities || [],
                      imageRefs: data.image_refs || [],
                      thinking: data.thinking || null,
                      performance: data.performance as RAGPerformanceMetrics | undefined,
                      agentSteps: localSteps, // include synced steps directly in finalMessage
                      timestamp: new Date().toISOString(),
                    };
                    break;
                  }

                  case "error":
                    setError(data.message || "未知错误");
                    setStatus("error");
                    syncUpdateSteps((prev) => markActiveError(prev));
                    break;
                }
              } catch {
                // Ignore malformed JSON
              }
            }
          }
        }

        readerRef.current = null;
        setStatus("idle");
        setIsStreaming(false);

        return finalMessage;
      } catch (err) {
        readerRef.current = null;
        if ((err as Error).name === "AbortError") {
          // User cancelled — don't set error
          return null;
        }
        const msg = (err as Error).message || "Stream failed";
        setError(msg);
        setStatus("error");
        setIsStreaming(false);
        syncUpdateSteps((prev) => markActiveError(prev));
        return null;
      }
    },
    [workspaceId, onToken, onThinkingToken],
  );

  return {
    status,
    streamingContent,
    thinkingText,
    pendingSources,
    pendingImages,
    error,
    isStreaming,
    agentSteps,
    sendMessage,
    cancel,
    reset,
  };
}
