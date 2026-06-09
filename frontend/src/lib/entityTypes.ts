const ENTITY_TYPE_LABELS: Record<string, string> = {
  person: "人物",
  organization: "组织",
  organisation: "组织",
  company: "公司",
  institution: "机构",
  product: "产品",
  model: "模型",
  artifact: "产物",
  method: "方法",
  content: "内容",
  dataset: "数据集",
  data: "数据",
  paper: "论文",
  algorithm: "算法",
  framework: "框架",
  task: "任务",
  metric: "指标",
  location: "地点",
  country: "国家",
  event: "事件",
  concept: "概念",
  technology: "技术",
  financial_metric: "财务指标",
  date: "日期",
  regulation: "法规",
  policy: "政策",
  law: "法律",
  project: "项目",
  document: "文档",
  unknown: "未知",
};

const ENTITY_TYPE_COLORS: Record<string, string> = {
  person: "#8b5cf6",
  organization: "#2563eb",
  organisation: "#2563eb",
  company: "#1d4ed8",
  institution: "#3b82f6",
  product: "#06b6d4",
  model: "#7c3aed",
  artifact: "#78716c",
  method: "#0ea5e9",
  content: "#64748b",
  dataset: "#84cc16",
  paper: "#475569",
  algorithm: "#0284c7",
  framework: "#2563eb",
  task: "#a855f7",
  metric: "#10b981",
  location: "#f59e0b",
  country: "#d97706",
  event: "#f97316",
  concept: "#14b8a6",
  technology: "#6366f1",
  financial_metric: "#059669",
  date: "#ec4899",
  regulation: "#ef4444",
  policy: "#e11d48",
  law: "#dc2626",
  project: "#d946ef",
  document: "#64748b",
  unknown: "#94a3b8",
};

export function normalizeEntityType(type: string): string {
  return type.trim().toLocaleLowerCase().replace(/[\s-]+/g, "_");
}

/** Converts backend entity-type identifiers to Chinese UI labels. */
export function getEntityTypeLabel(type: string): string {
  const normalized = normalizeEntityType(type);
  return ENTITY_TYPE_LABELS[normalized] ?? type;
}

/** Returns a stable, non-black color for both known and custom entity types. */
export function getEntityTypeColor(type: string): string {
  const normalized = normalizeEntityType(type) || "unknown";
  const configured = ENTITY_TYPE_COLORS[normalized];
  if (configured) return configured;

  let hash = 0;
  for (let index = 0; index < normalized.length; index += 1) {
    hash = ((hash << 5) - hash + normalized.charCodeAt(index)) | 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 68% 52%)`;
}
