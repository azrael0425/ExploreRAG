/**
 * ThinkingTimeline — Vertical timeline showing agent processing steps.
 *
 * Two modes:
 * - "live" — during streaming: always expanded, active step has spinner
 * - "embedded" — after complete: collapsed summary, click to expand
 */

import { useState, useRef, useEffect } from "react";
import {
  Brain,
  CheckCircle2,
  Loader2,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { AgentStep, AgentStepType } from "@/types";

// ---------------------------------------------------------------------------
// Step Configuration
// ---------------------------------------------------------------------------

interface StepConfig {
  label: string;
}

const STEP_CONFIG: Record<AgentStepType, StepConfig> = {
  analyzing: { label: "正在分析" },
  understood: { label: "已理解" },
  retrieving: { label: "正在检索" },
  sources_found: { label: "已找到来源" },
  generating: { label: "正在生成" },
  attachment: { label: "正在处理附件" },
  done: { label: "已完成" },
  error: { label: "出错" },
};

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

/**
 * Agent steps are also stored with chat history, so translate the old English
 * details here rather than relying on newly streamed events alone.
 */
function getStepDetail(step: AgentStep): string {
  switch (step.step) {
    case "analyzing":
      return "正在分析你的问题…";
    case "understood":
      return "已理解问题";
    case "retrieving":
      return "正在检索知识库…";
    case "sources_found": {
      const sourceCount = step.sourceCount ?? 0;
      const imageText = step.imageCount ? `，关联 ${step.imageCount} 张图片` : "";
      return `已找到 ${sourceCount} 个来源${imageText}`;
    }
    case "generating":
      return "正在生成回答…";
    case "attachment":
      return step.detail || "正在处理临时附件…";
    case "done":
      return step.durationMs ? `已完成，用时 ${formatMs(step.durationMs)}` : "已完成";
    case "error":
      return "处理出错";
  }
}

// ---------------------------------------------------------------------------
// LiveTimer — updates every 100ms for active steps
// ---------------------------------------------------------------------------

function LiveTimer({ startTimestamp }: { startTimestamp: number }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const iv = setInterval(() => setElapsed(Date.now() - startTimestamp), 100);
    return () => clearInterval(iv);
  }, [startTimestamp]);

  return (
    <span className="text-[11px] font-mono tabular-nums text-primary/80">
      {formatMs(elapsed)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// ThinkingLogSection — collapsible full thinking log (embedded mode, post-stream)
// ---------------------------------------------------------------------------

function ThinkingLogSection({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-1.5">
      <button
        onClick={() => setExpanded((p) => !p)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground/70 hover:text-muted-foreground transition-colors"
      >
        <Brain className="w-2.5 h-2.5" />
        <span>{expanded ? "隐藏" : "显示"}思考记录</span>
        <ChevronDown
          className={cn(
            "w-2.5 h-2.5 transition-transform",
            expanded && "rotate-180",
          )}
        />
      </button>
      {expanded && (
          <div>
            <div
              className={cn(
                "mt-1 ml-1 text-[11px] leading-relaxed text-muted-foreground/80 italic",
                "max-h-[200px] overflow-y-auto",
                "border-l border-border/40 pl-2",
                "whitespace-pre-wrap break-words",
              )}
            >
              {text}
            </div>
          </div>
        )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StepNode — single step in the timeline
// ---------------------------------------------------------------------------

interface StepNodeProps {
  step: AgentStep;
  isLive: boolean;
}

function StepNode({ step, isLive }: StepNodeProps) {
  const isActive = step.status === "active";
  const isError = step.status === "error";
  const isCompleted = step.status === "completed";

  return (
    <div>
      {/* Content */}
      <div className="flex-1 min-w-0 pb-2.5">
        <div className="flex items-center gap-1.5 min-h-[18px]">
          <span
            className={cn(
              "text-xs leading-tight",
              isActive && "text-foreground font-medium",
              isCompleted && step.step !== "done" && "text-muted-foreground",
              step.step === "done" && "text-emerald-500 font-medium",
              isError && "text-destructive font-medium",
            )}
          >
            {getStepDetail(step)}
          </span>

          <span className="ml-auto flex-shrink-0">
            {isActive && isLive ? (
              <LiveTimer startTimestamp={step.timestamp} />
            ) : step.durationMs != null && step.durationMs > 0 ? (
              <span className="text-[11px] font-mono tabular-nums text-muted-foreground/70">
                {formatMs(step.durationMs)}
              </span>
            ) : null}
          </span>
        </div>

        {/* Thinking text: during active streaming, the inline preview in MessageBubble
            handles display. After completion, show collapsible log here. */}
        {step.step === "analyzing" && step.thinkingText && !isActive && (
          <ThinkingLogSection text={step.thinkingText} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TimelineSummary — collapsed 1-line summary for embedded mode
// ---------------------------------------------------------------------------

function buildSummary(steps: AgentStep[]): string {
  const sourcesStep = steps.find((s) => s.step === "sources_found");
  const doneStep = steps.find((s) => s.step === "done");

  const parts: string[] = [];

  if (sourcesStep) {
    let sourceText = `${sourcesStep.sourceCount || 0} 个来源`;
    if (sourcesStep.imageCount) {
      sourceText += ` + ${sourcesStep.imageCount} 张图片`;
    }
    parts.push(sourceText);
  }

  if (doneStep?.durationMs) {
    parts.push(formatMs(doneStep.durationMs));
  } else if (doneStep) {
    // Extract duration from detail if available
    const match = doneStep.detail.match(/[\d.]+[sm]/);
    if (match) parts.push(match[0]);
  }

  const activeStep = steps.find((s) => s.status === "active");

  if (parts.length === 0) {
    // Still in progress, show active step label
    if (activeStep) {
      const cfg = STEP_CONFIG[activeStep.step];
      return cfg ? `${cfg.label}...` : "正在处理...";
    }
    return "已完成";
  }
  if (sourcesStep) {
    const suffix = parts[1] ? `，耗时 ${parts[1]}` : activeStep ? "，正在生成..." : "";
    return `已找到 ${parts[0]}${suffix}`;
  }
  return `已完成，耗时 ${parts[0]}`;
}

// ---------------------------------------------------------------------------
// ThinkingTimeline — main export
// ---------------------------------------------------------------------------

interface ThinkingTimelineProps {
  steps: AgentStep[];
  mode: "live" | "embedded";
  className?: string;
  /** When true, auto-collapse the timeline (used when answer starts streaming). */
  autoCollapse?: boolean;
}

export function ThinkingTimeline({
  steps,
  mode,
  className,
  autoCollapse = false,
}: ThinkingTimelineProps) {
  // Live mode starts expanded; embedded mode (completed message) starts collapsed
  const [expanded, setExpanded] = useState(mode === "live");
  const hasAutoCollapsedRef = useRef(false);
  const prevModeRef = useRef(mode);

  // Live mode without autoCollapse → expanded
  // When autoCollapse kicks in → collapse once
  // When mode transitions live→embedded → stay collapsed
  useEffect(() => {
    if (autoCollapse && !hasAutoCollapsedRef.current) {
      hasAutoCollapsedRef.current = true;
      setExpanded(false);
    }
  }, [autoCollapse]);

  // When mode changes from live→embedded (streaming finished),
  // keep current collapsed state — do NOT re-expand
  useEffect(() => {
    prevModeRef.current = mode;
  }, [mode]);

  if (steps.length === 0) return null;

  // Collapsed summary — styled like ThinkingPanel header for visibility
  const isStillActive = steps.some((s) => s.status === "active");
  if (!expanded) {
    return (
      <div className={cn("rounded-md border border-border/60 bg-background overflow-hidden", className)}>
        <button
          onClick={() => setExpanded(true)}
          className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-primary/80 hover:text-primary transition-colors"
        >
          {isStillActive ? (
            <Loader2 className="w-3 h-3 animate-spin text-primary/80 flex-shrink-0" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-emerald-500/80 flex-shrink-0" />
          )}
          <span className="flex-1 text-left">{buildSummary(steps)}</span>
          <ChevronDown className="w-3 h-3 flex-shrink-0" />
        </button>
      </div>
    );
  }

  // Expanded — wrap in styled container for embedded mode
  const isEmbedded = mode === "embedded" || autoCollapse;

  return (
    <div
      className={cn(
        "relative",
        isEmbedded && "rounded-md border border-border/60 bg-background overflow-hidden",
        className,
      )}
    >
      {/* Header / collapse button for embedded mode */}
      {isEmbedded && (
        <button
          onClick={() => setExpanded(false)}
          className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-primary/80 hover:text-primary transition-colors border-b border-border/40"
        >
          <CheckCircle2 className="w-3 h-3 text-emerald-500/80 flex-shrink-0" />
          <span className="flex-1 text-left">{buildSummary(steps)}</span>
          <ChevronDown className="w-3 h-3 flex-shrink-0 rotate-180" />
        </button>
      )}

      <div className={cn(isEmbedded && "px-2.5 py-2")}>
        {steps.map((step) => (
          <StepNode
            key={step.id}
            step={step}
            isLive={mode === "live"}
          />
        ))}
      </div>
    </div>
  );
}
