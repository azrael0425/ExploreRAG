import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  ArrowLeft,
  Beaker,
  Check,
  Download,
  FilePlus2,
  FlaskConical,
  Loader2,
  Play,
  Save,
  Snowflake,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Upload,
} from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

type MetricSummary = {
  avg: number | null;
  evaluated_count: number;
  skipped_count: number;
  error_count: number;
};

type PerformanceSummary = {
  avg: number | null;
  p50: number | null;
  p95: number | null;
  count: number;
};

type EvalCase = {
  id: number;
  workspace_id: number;
  question: string;
  reference_answer?: string | null;
  reference_chunk_ids: string[];
  reference_contexts: string[];
  tags: string[];
  status: "draft" | "active" | "archived";
  source: "manual" | "ai" | "production";
  dataset_name: string;
  dataset_version: number;
  split: "dev" | "test";
  is_frozen: boolean;
  category: string;
  difficulty: "easy" | "medium" | "hard";
  expected_behavior: "answer" | "refuse";
  review_status: "draft" | "approved" | "rejected";
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  reference_entity_names: string[];
  reference_relationships: Record<string, unknown>[];
};

type EvalRun = {
  id: number;
  workspace_id: number;
  status: string;
  run_type: "retrieval" | "fast" | "full";
  name?: string | null;
  experiment_id?: string | null;
  variant: "A" | "B" | "C" | "D" | "custom";
  dataset_name?: string | null;
  dataset_version?: number | null;
  dataset_split?: string | null;
  case_ids: number[];
  config?: Record<string, unknown> | null;
  metrics_summary?: {
    case_count?: number;
    pass_rate?: number | null;
    verdicts?: Record<string, number>;
    metrics?: Record<string, MetricSummary>;
    performance?: Record<string, PerformanceSummary>;
    experiment_valid?: boolean;
    baseline_comparison?: {
      status: string;
      metric_deltas?: Record<string, { delta: number; ci95?: number[] | null }>;
    };
  } | null;
  is_baseline: boolean;
  baseline_run_id?: number | null;
  created_at: string;
  error_message?: string | null;
};

type EvalResult = {
  id: number;
  case_id: number;
  question: string;
  answer?: string | null;
  reference_answer?: string | null;
  retrieved_contexts: string[];
  metrics: Record<string, number | null>;
  metric_status: Record<string, string>;
  failure_types: string[];
  baseline_delta: Record<string, number>;
  retrieval_trace: {
    reranker?: { requested?: boolean; applied?: boolean; status?: string };
    knowledge_graph?: { requested?: boolean; applied?: boolean; status?: string };
  };
  verdict: string;
  error_message?: string | null;
};

type Feedback = {
  message_id: string;
  rating: -1 | 1;
  question?: string | null;
  answer: string;
  comment?: string | null;
  source_ratings: Record<string, -1 | 1>;
  corrected_answer?: string | null;
  reference_chunk_ids: string[];
  failure_types: string[];
  review_status?: string | null;
  promoted_case_id?: number | null;
};

const CATEGORIES = [
  "single_hop",
  "multi_hop",
  "cross_document",
  "table_numeric",
  "unanswerable",
  "citation",
  "multi_turn",
];

const FAILURE_TYPES = [
  "knowledge_gap",
  "retrieval_miss",
  "rerank_error",
  "graph_miss",
  "graph_noise",
  "generation_error",
  "citation_error",
  "unanswerable_error",
];

const statusStyle: Record<string, string> = {
  active: "bg-emerald-500/10 text-emerald-700",
  approved: "bg-emerald-500/10 text-emerald-700",
  draft: "bg-amber-500/10 text-amber-700",
  rejected: "bg-destructive/10 text-destructive",
  completed: "bg-emerald-500/10 text-emerald-700",
  running: "bg-sky-500/10 text-sky-700",
  queued: "bg-amber-500/10 text-amber-700",
  failed: "bg-destructive/10 text-destructive",
  pass: "bg-emerald-500/10 text-emerald-700",
  fail: "bg-destructive/10 text-destructive",
  needs_review: "bg-amber-500/10 text-amber-700",
};

const statusLabel: Record<string, string> = {
  active: "已启用",
  approved: "已审核",
  draft: "待审核",
  rejected: "已拒绝",
  archived: "已归档",
  manual: "人工创建",
  ai: "AI 候选",
  production: "线上反馈",
  completed: "已完成",
  running: "运行中",
  queued: "排队中",
  failed: "失败",
  pass: "通过",
  fail: "未通过",
  needs_review: "需复核",
  retrieval: "仅检索",
  fast: "快速问答",
  full: "完整 Ragas",
};

function Badge({ value }: { value: string }) {
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-medium", statusStyle[value] || "bg-muted text-muted-foreground")}>
      {statusLabel[value] || value}
    </span>
  );
}

function CaseCard({
  item,
  reviewer,
  perform,
}: {
  item: EvalCase;
  reviewer: string;
  perform: (name: string, action: () => Promise<unknown>) => Promise<void>;
}) {
  const [answer, setAnswer] = useState(item.reference_answer || "");
  const [chunkIds, setChunkIds] = useState(item.reference_chunk_ids.join(", "));
  const [category, setCategory] = useState(item.category);
  const [split, setSplit] = useState(item.split);
  const [entityNames, setEntityNames] = useState(item.reference_entity_names.join(", "));
  const [relationships, setRelationships] = useState(JSON.stringify(item.reference_relationships, null, 2));

  useEffect(() => {
    setAnswer(item.reference_answer || "");
    setChunkIds(item.reference_chunk_ids.join(", "));
    setCategory(item.category);
    setSplit(item.split);
    setEntityNames(item.reference_entity_names.join(", "));
    setRelationships(JSON.stringify(item.reference_relationships, null, 2));
  }, [item]);

  const parsedChunkIds = chunkIds.split(/[,\n]/).map((value) => value.trim()).filter(Boolean);
  const save = () => perform(`save-case-${item.id}`, async () => {
    const parsedRelationships = relationships.trim() ? JSON.parse(relationships) : [];
    if (!Array.isArray(parsedRelationships)) throw new Error("Gold 关系必须是 JSON 数组");
    return api.patch(`/evaluations/cases/${item.id}`, {
      reference_answer: answer.trim() || null,
      reference_chunk_ids: parsedChunkIds,
      reference_entity_names: entityNames.split(/[,\n]/).map((value) => value.trim()).filter(Boolean),
      reference_relationships: parsedRelationships,
      category,
      split,
      expected_behavior: category === "unanswerable" ? "refuse" : "answer",
    });
  });
  const approve = () => {
    if (!reviewer.trim()) return Promise.reject(new Error("请先填写审核人"));
    return api.post(`/evaluations/workspaces/${item.workspace_id}/cases/review`, {
      case_ids: [item.id],
      review_status: "approved",
      reviewer: reviewer.trim(),
      activate: true,
      freeze: split === "test",
    });
  };

  return (
    <details className="rounded-lg border p-3" open={item.review_status === "draft" && item.source !== "manual"}>
      <summary className="flex cursor-pointer list-none items-start justify-between gap-2">
        <div>
          <p className="text-sm leading-relaxed">{item.question}</p>
          <p className="mt-1 text-[10px] text-muted-foreground">
            {item.dataset_name} v{item.dataset_version} · {item.split} · {item.category}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-1">
          {item.is_frozen && <Snowflake className="h-3.5 w-3.5 text-sky-600" />}
          <Badge value={item.review_status} />
          <Badge value={item.source} />
        </div>
      </summary>
      <div className="mt-3 space-y-2 border-t pt-3">
        {item.reference_contexts.length > 0 && (
          <details className="rounded bg-muted/40 p-2">
            <summary className="cursor-pointer text-[11px] text-muted-foreground">查看 Gold 上下文 ({item.reference_contexts.length})</summary>
            <div className="mt-2 max-h-44 space-y-2 overflow-auto text-[11px] leading-relaxed">
              {item.reference_contexts.map((context, index) => <p key={index} className="whitespace-pre-wrap">{context}</p>)}
            </div>
          </details>
        )}
        <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} rows={3} disabled={item.is_frozen} placeholder="人工修订后的参考答案" className="w-full rounded-md border bg-background p-2 text-xs disabled:opacity-60" />
        <input value={chunkIds} onChange={(event) => setChunkIds(event.target.value)} disabled={item.is_frozen} placeholder="Gold chunk_id，逗号分隔" className="w-full rounded-md border bg-background p-2 font-mono text-[11px] disabled:opacity-60" />
        <input value={entityNames} onChange={(event) => setEntityNames(event.target.value)} disabled={item.is_frozen} placeholder="Gold 实体名称，逗号分隔（关系/多跳题必填）" className="w-full rounded-md border bg-background p-2 text-[11px] disabled:opacity-60" />
        <textarea value={relationships} onChange={(event) => setRelationships(event.target.value)} rows={2} disabled={item.is_frozen} placeholder='Gold 关系 JSON 数组，例如 [{"source":"A","relation":"uses","target":"B"}]' className="w-full rounded-md border bg-background p-2 font-mono text-[10px] disabled:opacity-60" />
        <div className="flex flex-wrap items-center gap-2">
          <select value={category} onChange={(event) => setCategory(event.target.value)} disabled={item.is_frozen} className="rounded border bg-background px-2 py-1.5 text-xs">
            {CATEGORIES.map((value) => <option key={value}>{value}</option>)}
          </select>
          <select value={split} onChange={(event) => setSplit(event.target.value as "dev" | "test")} disabled={item.is_frozen} className="rounded border bg-background px-2 py-1.5 text-xs">
            <option value="dev">dev</option>
            <option value="test">test（审核后冻结）</option>
          </select>
          <span className="text-[10px] text-muted-foreground">{parsedChunkIds.length} 个 Gold chunk</span>
          <div className="ml-auto flex gap-2">
            {item.is_frozen ? (
              <button onClick={() => perform(`unfreeze-${item.id}`, () => api.patch(`/evaluations/cases/${item.id}`, { is_frozen: false }))} className="rounded border px-2 py-1 text-[11px] text-sky-700">显式解冻</button>
            ) : (
              <>
                <button onClick={save} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-[11px]"><Save className="h-3 w-3" />保存修订</button>
                <button onClick={() => perform(`approve-${item.id}`, approve)} className="inline-flex items-center gap-1 rounded bg-primary px-2 py-1 text-[11px] text-primary-foreground"><Check className="h-3 w-3" />审核通过</button>
              </>
            )}
          </div>
        </div>
        {item.reviewed_by && <p className="text-[10px] text-muted-foreground">审核：{item.reviewed_by} · {item.reviewed_at ? new Date(item.reviewed_at).toLocaleString() : ""}</p>}
      </div>
    </details>
  );
}

function FeedbackCard({
  item,
  workspaceId,
  perform,
}: {
  item: Feedback;
  workspaceId: string;
  perform: (name: string, action: () => Promise<unknown>) => Promise<void>;
}) {
  const [answer, setAnswer] = useState(item.corrected_answer || "");
  const [chunkIds, setChunkIds] = useState(item.reference_chunk_ids.join(", "));
  const [failureType, setFailureType] = useState(item.failure_types[0] || (item.rating === -1 ? "generation_error" : ""));
  const saveReview = () => api.post(`/evaluations/workspaces/${workspaceId}/messages/${item.message_id}/feedback`, {
    rating: item.rating,
    comment: item.comment || null,
    source_ratings: item.source_ratings,
    corrected_answer: answer.trim() || null,
    reference_chunk_ids: chunkIds.split(/[,\n]/).map((value) => value.trim()).filter(Boolean),
    failure_types: failureType ? [failureType] : [],
    review_status: "reviewed",
  });

  return (
    <details className="rounded-lg border p-2">
      <summary className="flex cursor-pointer list-none justify-between gap-2">
        <p className="line-clamp-1 text-xs">{item.question || "未关联历史问题"}</p>
        <div className="flex items-center gap-1">
          {item.rating === 1 ? <ThumbsUp className="h-3.5 w-3.5 text-emerald-600" /> : <ThumbsDown className="h-3.5 w-3.5 text-destructive" />}
          {item.review_status && <Badge value={item.review_status} />}
        </div>
      </summary>
      <p className="mt-2 line-clamp-3 text-[10px] text-muted-foreground">原回答：{item.answer}</p>
      <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} rows={3} placeholder="人工修订后的 Gold Answer（正反馈也需确认）" className="mt-2 w-full rounded border p-2 text-xs" />
      <input value={chunkIds} onChange={(event) => setChunkIds(event.target.value)} placeholder="人工确认的 Gold chunk_id" className="mt-2 w-full rounded border p-2 font-mono text-[11px]" />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <select value={failureType} onChange={(event) => setFailureType(event.target.value)} className="rounded border bg-background px-2 py-1 text-[11px]">
          <option value="">无失败（正反馈）</option>
          {FAILURE_TYPES.map((value) => <option key={value}>{value}</option>)}
        </select>
        <button onClick={() => perform(`feedback-save-${item.message_id}`, saveReview)} className="rounded border px-2 py-1 text-[11px]">保存人工复核</button>
        <button disabled={item.review_status !== "reviewed" || !!item.promoted_case_id} onClick={() => perform(`promote-${item.message_id}`, () => api.post(`/evaluations/workspaces/${workspaceId}/messages/${item.message_id}/promote`))} className="rounded bg-primary px-2 py-1 text-[11px] text-primary-foreground disabled:opacity-40">
          {item.promoted_case_id ? `已进入案例 #${item.promoted_case_id}` : "晋升为案例草稿"}
        </button>
      </div>
    </details>
  );
}

export function EvaluationPage() {
  const { workspaceId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [referenceIds, setReferenceIds] = useState("");
  const [manualCategory, setManualCategory] = useState("single_hop");
  const [manualSplit, setManualSplit] = useState<"dev" | "test">("dev");
  const [generateCount, setGenerateCount] = useState(8);
  const [generateCategory, setGenerateCategory] = useState("single_hop");
  const [generateSplit, setGenerateSplit] = useState<"dev" | "test">("dev");
  const [generationSeed, setGenerationSeed] = useState(0);
  const [jsonl, setJsonl] = useState("");
  const [reviewer, setReviewer] = useState(() => localStorage.getItem("evaluation-reviewer") || "");
  const [caseFilter, setCaseFilter] = useState<"all" | "draft" | "approved">("draft");
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [runType, setRunType] = useState<"retrieval" | "fast" | "full">("retrieval");
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem("evaluation-reviewer", reviewer);
  }, [reviewer]);

  const key = ["evaluation", workspaceId];
  const overview = useQuery({
    queryKey: [...key, "overview"],
    queryFn: () => api.get<{ active_case_count: number; draft_case_count: number; feedback_count: number }>(`/evaluations/workspaces/${workspaceId}/overview`),
    enabled: !!workspaceId,
  });
  const cases = useQuery({
    queryKey: [...key, "cases"],
    queryFn: () => api.get<{ items: EvalCase[] }>(`/evaluations/workspaces/${workspaceId}/cases`),
    enabled: !!workspaceId,
  });
  const runs = useQuery({
    queryKey: [...key, "runs"],
    queryFn: () => api.get<EvalRun[]>(`/evaluations/workspaces/${workspaceId}/runs`),
    enabled: !!workspaceId,
    refetchInterval: (query) => query.state.data?.some((run) => run.status === "queued" || run.status === "running") ? 2500 : false,
  });
  const feedback = useQuery({
    queryKey: [...key, "feedback"],
    queryFn: () => api.get<Feedback[]>(`/evaluations/workspaces/${workspaceId}/feedback`),
    enabled: !!workspaceId,
  });
  const selectedRun = selectedRunId ?? runs.data?.[0]?.id ?? null;
  const runDetail = useQuery({
    queryKey: [...key, "run", selectedRun],
    queryFn: () => api.get<EvalRun & { results: EvalResult[] }>(`/evaluations/runs/${selectedRun}`),
    enabled: !!selectedRun,
    refetchInterval: runs.data?.find((run) => run.id === selectedRun)?.status === "running" ? 2500 : false,
  });

  const approvedCases = useMemo(
    () => cases.data?.items.filter((item) => item.status === "active" && item.review_status === "approved") ?? [],
    [cases.data],
  );
  const visibleCases = useMemo(
    () => cases.data?.items.filter((item) => caseFilter === "all" || item.review_status === caseFilter) ?? [],
    [cases.data, caseFilter],
  );
  const datasetCounts = useMemo(() => {
    const values = { dev: 0, test: 0, frozen: 0, approved: 0 };
    for (const item of cases.data?.items || []) {
      values[item.split] += 1;
      if (item.is_frozen) values.frozen += 1;
      if (item.review_status === "approved") values.approved += 1;
    }
    return values;
  }, [cases.data]);

  const refresh = async () => queryClient.invalidateQueries({ queryKey: key });
  const perform = async (name: string, action: () => Promise<unknown>) => {
    setBusy(name);
    try {
      await action();
      await refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "操作失败");
    } finally {
      setBusy(null);
    }
  };

  const createCase = () => perform("create", async () => {
    if (!question.trim()) throw new Error("请输入问题");
    await api.post(`/evaluations/workspaces/${workspaceId}/cases`, {
      question: question.trim(),
      reference_answer: referenceAnswer.trim() || null,
      reference_chunk_ids: referenceIds.split(/[,\n]/).map((id) => id.trim()).filter(Boolean),
      status: "draft",
      source: "manual",
      dataset_name: "explorerag_core",
      dataset_version: 1,
      split: manualSplit,
      category: manualCategory,
      expected_behavior: manualCategory === "unanswerable" ? "refuse" : "answer",
    });
    setQuestion("");
    setReferenceAnswer("");
    setReferenceIds("");
    toast.success("已创建待审核案例");
  });

  const createRun = () => perform(`run-${runType}`, async () => {
    const run = await api.post<EvalRun>(`/evaluations/workspaces/${workspaceId}/runs`, {
      case_ids: approvedCases.map((item) => item.id),
      run_type: runType,
      name: `${statusLabel[runType]} · ${new Date().toLocaleString()}`,
      top_k: 4,
      prefetch_k: 20,
      retrieval_mode: "hybrid",
      warmup_queries: 1,
      case_order_seed: 20260810,
    });
    setSelectedRunId(run.id);
    toast.success("评测已进入队列");
  });

  const createAblation = () => perform("ablation", async () => {
    const response = await api.post<{ runs: EvalRun[] }>(`/evaluations/workspaces/${workspaceId}/experiments/ablation`, {
      case_ids: approvedCases.map((item) => item.id),
      run_type: runType,
      name: `2×2 消融 · ${new Date().toLocaleString()}`,
      top_k: 4,
      prefetch_k: 20,
      retrieval_mode: "hybrid",
      warmup_queries: 1,
      case_order_seed: 20260810,
      variant_order: ["A", "B", "C", "D"],
    });
    setSelectedRunId(response.runs[0]?.id ?? null);
    toast.success("A/B/C/D 四组消融已按顺序入队");
  });

  const summary = runDetail.data?.metrics_summary;

  return (
    <main className="min-h-full bg-muted/20 px-4 py-5 md:px-7">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate(`/knowledge-bases/${workspaceId}`)} className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground" title="返回知识库"><ArrowLeft className="h-4 w-4" /></button>
            <div>
              <h1 className="flex items-center gap-2 text-xl font-semibold"><Beaker className="h-5 w-5 text-primary" />RAG 评测与知识飞轮</h1>
              <p className="text-xs text-muted-foreground">版本化数据集、2×2 消融、配对回归与人工审核反馈闭环</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="rounded-lg border bg-background px-3 py-2">候选 <b>{(cases.data?.items.length ?? 0) - datasetCounts.approved}</b></span>
            <span className="rounded-lg border bg-background px-3 py-2">已审核 <b>{datasetCounts.approved}</b></span>
            <span className="rounded-lg border bg-background px-3 py-2">dev/test <b>{datasetCounts.dev}/{datasetCounts.test}</b></span>
            <span className="rounded-lg border bg-background px-3 py-2">冻结 <b>{datasetCounts.frozen}</b></span>
            <span className="rounded-lg border bg-background px-3 py-2">反馈 <b>{overview.data?.feedback_count ?? "-"}</b></span>
          </div>
        </header>

        <section className="grid gap-5 xl:grid-cols-[1.25fr_1fr]">
          <div className="space-y-5">
            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div><h2 className="font-medium">评测集构建</h2><p className="text-[11px] text-muted-foreground">AI 只生成候选；修订 Gold Answer 和 Gold Chunk 后由具名审核人批准。</p></div>
                <label className="flex items-center gap-2 text-xs">审核人<input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="姓名或账号" className="w-32 rounded border px-2 py-1.5" /></label>
              </div>
              <details className="rounded-lg border p-3">
                <summary className="flex cursor-pointer items-center gap-1 text-sm font-medium"><FilePlus2 className="h-4 w-4" />新建人工案例</summary>
                <div className="mt-3 space-y-2">
                  <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={2} placeholder="用户问题" className="w-full rounded-md border p-2 text-sm" />
                  <textarea value={referenceAnswer} onChange={(event) => setReferenceAnswer(event.target.value)} rows={3} placeholder="参考答案" className="w-full rounded-md border p-2 text-sm" />
                  <input value={referenceIds} onChange={(event) => setReferenceIds(event.target.value)} placeholder="Gold chunk_id，逗号分隔" className="w-full rounded-md border p-2 text-sm" />
                  <div className="flex gap-2">
                    <select value={manualCategory} onChange={(event) => setManualCategory(event.target.value)} className="rounded border px-2 py-1.5 text-xs">{CATEGORIES.map((value) => <option key={value}>{value}</option>)}</select>
                    <select value={manualSplit} onChange={(event) => setManualSplit(event.target.value as "dev" | "test")} className="rounded border px-2 py-1.5 text-xs"><option value="dev">dev</option><option value="test">test</option></select>
                    <button disabled={busy === "create"} onClick={createCase} className="ml-auto rounded bg-primary px-3 py-1.5 text-xs text-primary-foreground">保存候选</button>
                  </div>
                </div>
              </details>
              <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border p-3">
                <Sparkles className="h-4 w-4 text-primary" />
                <input type="number" min={1} max={30} value={generateCount} onChange={(event) => setGenerateCount(Number(event.target.value))} className="w-16 rounded border p-1.5 text-xs" />
                <select value={generateCategory} onChange={(event) => setGenerateCategory(event.target.value)} className="rounded border px-2 py-1.5 text-xs">{CATEGORIES.map((value) => <option key={value}>{value}</option>)}</select>
                <select value={generateSplit} onChange={(event) => setGenerateSplit(event.target.value as "dev" | "test")} className="rounded border px-2 py-1.5 text-xs"><option value="dev">dev</option><option value="test">test</option></select>
                <input type="number" min={0} value={generationSeed} onChange={(event) => setGenerationSeed(Number(event.target.value))} className="w-20 rounded border p-1.5 text-xs" title="轮转采样 seed" />
                <button disabled={busy === "generate"} onClick={() => perform("generate", async () => {
                  await api.post(`/evaluations/workspaces/${workspaceId}/cases/generate`, {
                    count: generateCount,
                    categories: [generateCategory],
                    split: generateSplit,
                    seed: generationSeed,
                    dataset_name: "explorerag_core",
                    dataset_version: 1,
                  });
                  setGenerationSeed((value) => value + 1);
                  toast.success("AI 候选已进入待审核队列");
                })} className="rounded border px-3 py-1.5 text-xs">生成候选</button>
                <button onClick={() => api.downloadFile(`/evaluations/workspaces/${workspaceId}/cases/export`, `evaluation-cases-${workspaceId}.jsonl`)} className="ml-auto inline-flex items-center gap-1 rounded border px-2 py-1.5 text-xs"><Download className="h-3 w-3" />导出</button>
              </div>
            </section>

            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <h2 className="font-medium">案例审核队列</h2>
                <div className="flex gap-1">{(["draft", "approved", "all"] as const).map((value) => <button key={value} onClick={() => setCaseFilter(value)} className={cn("rounded border px-2 py-1 text-[11px]", caseFilter === value && "border-primary bg-primary/5 text-primary")}>{value === "all" ? "全部" : statusLabel[value]}</button>)}</div>
              </div>
              <div className="max-h-[720px] space-y-2 overflow-auto pr-1">
                {cases.isLoading ? <Loader2 className="m-4 h-4 w-4 animate-spin" /> : visibleCases.length ? visibleCases.map((item) => <CaseCard key={item.id} item={item} reviewer={reviewer} perform={perform} />) : <p className="py-8 text-center text-sm text-muted-foreground">当前筛选下没有案例。</p>}
              </div>
              <details className="mt-3 rounded-md border bg-muted/20 p-2">
                <summary className="flex cursor-pointer items-center gap-1 text-xs text-muted-foreground"><Upload className="h-3 w-3" />导入 JSONL</summary>
                <textarea value={jsonl} onChange={(event) => setJsonl(event.target.value)} rows={5} placeholder='每行一个案例 JSON，例如 {"question":"...","reference_answer":"..."}' className="mt-2 w-full rounded border bg-background p-2 font-mono text-[11px]" />
                <div className="mt-2 text-right"><button onClick={() => perform("import", async () => { await api.post(`/evaluations/workspaces/${workspaceId}/cases/import`, { jsonl }); setJsonl(""); })} disabled={!jsonl.trim() || !!busy} className="rounded border px-2 py-1 text-xs disabled:opacity-50">导入候选</button></div>
              </details>
            </section>
          </div>

          <div className="space-y-5">
            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between"><div><h2 className="font-medium">运行实验</h2><p className="text-[11px] text-muted-foreground">仅使用 active + approved 案例；正式结果建议选择冻结 test。</p></div><FlaskConical className="h-4 w-4 text-primary" /></div>
              <div className="flex flex-wrap items-center gap-2">
                <select value={runType} onChange={(event) => setRunType(event.target.value as "retrieval" | "fast" | "full")} className="rounded border bg-background px-2 py-1.5 text-xs"><option value="retrieval">仅检索（低成本）</option><option value="fast">快速问答</option><option value="full">完整 Ragas</option></select>
                <button onClick={createRun} disabled={!approvedCases.length || !!busy} className="inline-flex items-center gap-1 rounded border px-2 py-1.5 text-xs disabled:opacity-40"><Play className="h-3 w-3" />单次运行 ({approvedCases.length})</button>
                <button onClick={createAblation} disabled={!approvedCases.length || !!busy} className="inline-flex items-center gap-1 rounded bg-primary px-2 py-1.5 text-xs text-primary-foreground disabled:opacity-40"><FlaskConical className="h-3 w-3" />2×2 消融 A/B/C/D</button>
              </div>
            </section>

            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <h2 className="mb-3 font-medium">运行历史</h2>
              <div className="max-h-72 space-y-2 overflow-auto">
                {runs.data?.length ? runs.data.map((run) => (
                  <button key={run.id} onClick={() => setSelectedRunId(run.id)} className={cn("w-full rounded-lg border p-3 text-left hover:bg-muted/40", selectedRun === run.id && "border-primary/50 bg-primary/[0.03]")}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="line-clamp-1 text-xs font-medium">#{run.id} · {run.variant !== "custom" ? `${run.variant} · ` : ""}{run.name || statusLabel[run.run_type]}</span>
                      <div className="flex gap-1">{run.is_baseline && <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary">基线</span>}<Badge value={run.status} /></div>
                    </div>
                    <div className="mt-1 flex justify-between text-[10px] text-muted-foreground"><span>{new Date(run.created_at).toLocaleString()}</span><span>{run.metrics_summary?.case_count ?? run.case_ids.length} 案例 · {statusLabel[run.run_type]}</span></div>
                  </button>
                )) : <p className="py-5 text-center text-sm text-muted-foreground">尚未运行。</p>}
              </div>
            </section>

            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between"><h2 className="font-medium">运行结果</h2>{runDetail.data && <button onClick={() => perform(`baseline-${runDetail.data.id}`, () => api.post(`/evaluations/runs/${runDetail.data.id}/baseline`))} className="text-[11px] text-primary hover:underline">设为基线</button>}</div>
              {runDetail.isLoading ? <Loader2 className="m-4 h-4 w-4 animate-spin" /> : runDetail.data ? (
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-2 text-[11px]"><Badge value={runDetail.data.status} /><span>通过率 {summary?.pass_rate == null ? "-" : `${(summary.pass_rate * 100).toFixed(1)}%`}</span>{summary?.experiment_valid != null && <span className={summary.experiment_valid ? "text-emerald-700" : "text-destructive"}>组件指纹：{summary.experiment_valid ? "有效" : "无效/发生降级"}</span>}</div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {Object.entries(summary?.metrics || {}).map(([name, value]) => <div key={name} className="rounded-md bg-muted/50 px-2 py-1.5"><p className="truncate text-[10px] text-muted-foreground">{name} · n={value.evaluated_count}</p><p className="text-sm font-semibold">{value.avg == null ? "-" : value.avg.toFixed(3)}</p></div>)}
                  </div>
                  {summary?.performance && <div className="grid grid-cols-2 gap-2">{Object.entries(summary.performance).filter(([, value]) => value.p95 != null).map(([name, value]) => <div key={name} className="rounded border px-2 py-1.5 text-[10px]"><span className="text-muted-foreground">{name}</span><p>P50 {value.p50?.toFixed(0)} ms · P95 {value.p95?.toFixed(0)} ms</p></div>)}</div>}
                  {summary?.baseline_comparison?.metric_deltas && <div className="rounded border border-primary/20 bg-primary/[0.03] p-2"><p className="mb-1 text-[11px] font-medium">相对基线（paired bootstrap 95% CI）</p>{Object.entries(summary.baseline_comparison.metric_deltas).map(([name, delta]) => <p key={name} className="text-[10px]">{name}: {delta.delta >= 0 ? "+" : ""}{delta.delta.toFixed(4)} {delta.ci95 ? `[${delta.ci95.join(", ")}]` : ""}</p>)}</div>}
                  {runDetail.data.error_message && <p className="text-xs text-destructive">{runDetail.data.error_message}</p>}
                  <div className="max-h-[520px] space-y-2 overflow-auto pr-1">
                    {runDetail.data.results.map((result) => <details key={result.id} className="rounded-lg border p-2"><summary className="flex cursor-pointer list-none items-center justify-between gap-2"><span className="line-clamp-1 text-xs">{result.question}</span><Badge value={result.verdict} /></summary><p className="mt-2 whitespace-pre-wrap text-xs leading-relaxed">{result.answer || result.error_message || "仅检索运行"}</p><div className="mt-2 grid grid-cols-2 gap-1">{Object.entries(result.metrics).map(([name, value]) => <span key={name} className="rounded bg-muted px-1.5 py-1 text-[10px]">{name}: {value === null ? result.metric_status[name] : Number(value).toFixed(3)}</span>)}</div><div className="mt-2 flex flex-wrap gap-1 text-[10px]"><span>Reranker: {result.retrieval_trace.reranker?.status || "unknown"}</span><span>KG: {result.retrieval_trace.knowledge_graph?.status || "unknown"}</span>{result.failure_types.map((value) => <Badge key={value} value={value} />)}</div></details>)}
                  </div>
                </div>
              ) : <p className="py-5 text-center text-sm text-muted-foreground">选择运行记录查看结果。</p>}
            </section>

            <section className="rounded-xl border bg-background p-4 shadow-sm">
              <h2 className="mb-1 font-medium">线上反馈回流</h2>
              <p className="mb-3 text-[11px] text-muted-foreground">正反馈也不能直接成为 Gold；先修订答案、确认 Gold Chunk、归因，再晋升为草稿。</p>
              <div className="max-h-[420px] space-y-2 overflow-auto">{feedback.data?.length ? feedback.data.map((item) => <FeedbackCard key={item.message_id} item={item} workspaceId={workspaceId} perform={perform} />) : <p className="py-4 text-center text-xs text-muted-foreground">聊天中的点赞、点踩和引用评分会显示在这里。</p>}</div>
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}
