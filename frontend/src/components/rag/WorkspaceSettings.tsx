import { useState, useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Settings2,
  X,
  Save,
  RotateCcw,
  Cloud,
  Brain,
  HardDrive,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import type {
  KnowledgeBase,
  LLMMode,
  LLMRuntimeStatus,
  UpdateWorkspace,
  MetadataFieldDefinition,
  MetadataFieldType,
  MetadataSchema,
} from "@/types";

interface WorkspaceSettingsProps {
  workspace: KnowledgeBase;
  onSave: (data: UpdateWorkspace) => Promise<void>;
  open: boolean;
  onClose: () => void;
}

const FIELD_TYPE_LABELS: Record<MetadataFieldType, string> = {
  string: "文本",
  integer: "整数",
  number: "数值",
  boolean: "布尔值",
  date: "日期",
  datetime: "日期时间",
  enum: "单选",
  multi_enum: "多选",
};

export function WorkspaceSettings({ workspace, onSave, open, onClose }: WorkspaceSettingsProps) {
  const queryClient = useQueryClient();
  const [llmMode, setLlmMode] = useState<LLMMode>(workspace.llm_mode ?? "cloud");
  const [lightragAugmentationEnabled, setLightragAugmentationEnabled] = useState(
    workspace.lightrag_augmentation_enabled ?? false,
  );
  const [saving, setSaving] = useState(false);
  const [metadataSchema, setMetadataSchema] = useState<MetadataSchema>(workspace.metadata_schema ?? { version: 1, fields: [] });
  const [fieldKey, setFieldKey] = useState("");
  const [fieldLabel, setFieldLabel] = useState("");
  const [fieldType, setFieldType] = useState<MetadataFieldType>("string");
  const [fieldOptions, setFieldOptions] = useState("");
  const [fieldRequired, setFieldRequired] = useState(false);
  const [fieldFilterable, setFieldFilterable] = useState(true);
  const [fieldSemantic, setFieldSemantic] = useState(false);

  const { data: llmStatus, isFetching: statusLoading } = useQuery({
    queryKey: ["workspace-llm-status", workspace.id, llmMode],
    queryFn: () => api.get<LLMRuntimeStatus>(`/workspaces/${workspace.id}/llm-status?mode=${llmMode}`),
    enabled: open,
    retry: false,
    refetchInterval: open && llmMode === "local" ? 10_000 : false,
  });

  useEffect(() => {
    setLlmMode(workspace.llm_mode ?? "cloud");
  }, [workspace.llm_mode]);

  useEffect(() => {
    setLightragAugmentationEnabled(workspace.lightrag_augmentation_enabled ?? false);
  }, [workspace.lightrag_augmentation_enabled]);

  useEffect(() => {
    setMetadataSchema(workspace.metadata_schema ?? { version: 1, fields: [] });
  }, [workspace.metadata_schema]);

  const schemaChanged = JSON.stringify(metadataSchema) !== JSON.stringify(workspace.metadata_schema ?? { version: 1, fields: [] });
  const lightragAvailable = workspace.lightrag_available ?? true;
  const hasChanges =
    llmMode !== (workspace.llm_mode ?? "cloud") ||
    lightragAugmentationEnabled !== (workspace.lightrag_augmentation_enabled ?? false) ||
    schemaChanged;

  const addMetadataField = useCallback(() => {
    const key = fieldKey.trim().toLowerCase();
    const label = fieldLabel.trim();
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(key)) {
      toast.error("字段标识只能使用小写字母、数字和下划线");
      return;
    }
    if (!label) {
      toast.error("请输入字段显示名称");
      return;
    }
    if (metadataSchema.fields.some((field) => field.key === key)) {
      toast.error("该字段标识已存在");
      return;
    }
    const options = fieldOptions.split(",").map((value) => value.trim()).filter(Boolean);
    if ((fieldType === "enum" || fieldType === "multi_enum") && !options.length) {
      toast.error("单选和多选字段至少需要一个选项");
      return;
    }
    const field: MetadataFieldDefinition = {
      key,
      label,
      type: fieldType,
      required: fieldRequired,
      filterable: fieldFilterable,
      semantic: fieldSemantic,
      options: fieldType === "enum" || fieldType === "multi_enum" ? options : [],
    };
    setMetadataSchema((schema) => ({ ...schema, fields: [...schema.fields, field] }));
    setFieldKey("");
    setFieldLabel("");
    setFieldOptions("");
    setFieldType("string");
    setFieldRequired(false);
    setFieldFilterable(true);
    setFieldSemantic(false);
  }, [fieldKey, fieldLabel, fieldOptions, fieldType, fieldRequired, fieldFilterable, fieldSemantic, metadataSchema.fields]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const workspacePatch: UpdateWorkspace = {};
      if (llmMode !== (workspace.llm_mode ?? "cloud")) {
        workspacePatch.llm_mode = llmMode;
      }
      if (lightragAugmentationEnabled !== (workspace.lightrag_augmentation_enabled ?? false)) {
        workspacePatch.lightrag_augmentation_enabled = lightragAugmentationEnabled;
      }
      if (Object.keys(workspacePatch).length > 0) {
        await onSave(workspacePatch);
      }
      if (schemaChanged) {
        const nextSchema = {
          ...metadataSchema,
          version: Math.max(workspace.metadata_schema?.version ?? 1, metadataSchema.version) + 1,
        };
        await api.put<MetadataSchema>(`/workspaces/${workspace.id}/metadata-schema`, nextSchema);
        queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      }
      toast.success("知识库设置已保存");
      onClose();
    } catch {
      toast.error("保存设置失败");
    } finally {
      setSaving(false);
    }
  }, [llmMode, lightragAugmentationEnabled, metadataSchema, onSave, onClose, queryClient, schemaChanged, workspace.id, workspace.llm_mode, workspace.lightrag_augmentation_enabled, workspace.metadata_schema]);

  if (!open) return null;

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-background/95 backdrop-blur-sm">
      <div className="flex flex-shrink-0 items-center justify-between border-b px-3 py-2">
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold">知识库设置</h2>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-7 w-7">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-3 py-3">
        <div className="space-y-2">
          <label className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <HardDrive className="h-3.5 w-3.5" />
            LLM 运行模式
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setLlmMode("cloud")}
              className={`rounded-md border p-2.5 text-left transition-colors ${llmMode === "cloud" ? "border-primary bg-primary/10" : "border-border hover:bg-muted/50"}`}
            >
              <div className="flex items-center gap-1.5 text-xs font-medium"><Cloud className="h-3.5 w-3.5" />云端模型</div>
              <p className="mt-1 text-[10px] text-muted-foreground">使用服务端配置的 DashScope 云端模型</p>
            </button>
            <button
              type="button"
              onClick={() => setLlmMode("local")}
              className={`rounded-md border p-2.5 text-left transition-colors ${llmMode === "local" ? "border-primary bg-primary/10" : "border-border hover:bg-muted/50"}`}
            >
              <div className="flex items-center gap-1.5 text-xs font-medium"><HardDrive className="h-3.5 w-3.5" />本地模型</div>
              <p className="mt-1 text-[10px] text-muted-foreground">数据不发送到云端，本地服务不可用时不会自动回退</p>
            </button>
          </div>
          <div className="flex items-start gap-1.5 rounded-md bg-muted/50 px-2 py-1.5 text-[10px]">
            {statusLoading ? <Loader2 className="mt-0.5 h-3 w-3 animate-spin text-muted-foreground" /> : llmStatus?.available ? <CheckCircle2 className="mt-0.5 h-3 w-3 text-emerald-500" /> : <AlertCircle className="mt-0.5 h-3 w-3 text-destructive" />}
            <span className="min-w-0 text-muted-foreground">
              {statusLoading ? "正在检查模型服务..." : llmStatus?.available ? `${llmStatus.provider} · ${llmStatus.model} 可用` : `模型服务不可用${llmStatus?.detail ? `：${llmStatus.detail}` : ""}`}
            </span>
          </div>
        </div>

        <div className="space-y-2 border-t pt-4">
          <div className="flex items-start justify-between gap-3 rounded-md border bg-muted/20 p-2.5">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Brain className="h-3.5 w-3.5" />
                LightRAG 回答增强
              </div>
              <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
                将知识图谱中的实体、关系和关联片段作为额外证据加入回答。关闭后仍会构建并显示知识图谱；启用后可能增加少量检索耗时。
              </p>
              {!lightragAvailable && (
                <p className="mt-1 text-[10px] text-destructive">
                  服务端未启用 LightRAG，当前设置不会生效。
                </p>
              )}
              {lightragAugmentationEnabled && lightragAvailable && (
                <p className="mt-1 text-[10px] text-muted-foreground">
                  指定文档或元数据筛选时，为避免跨范围图谱信息，本次查询会自动退回向量检索。
                </p>
              )}
            </div>
            <Switch
              checked={lightragAugmentationEnabled}
              onCheckedChange={setLightragAugmentationEnabled}
              disabled={!lightragAvailable}
              aria-label="启用 LightRAG 回答增强"
            />
          </div>
        </div>

        <div className="space-y-2 border-t pt-4">
          <div>
            <p className="text-xs font-medium text-muted-foreground">导入元数据字段</p>
            <p className="mt-1 text-[10px] text-muted-foreground">字段会在导入时校验，也可用于检索筛选。修改普通元数据无需重新建立索引。</p>
          </div>
          {metadataSchema.fields.length > 0 ? (
            <div className="space-y-1.5">
              {metadataSchema.fields.map((field) => (
                <div key={field.key} className="flex items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-xs">
                  <div className="min-w-0"><span className="font-medium">{field.label}</span><span className="ml-1 text-muted-foreground">{field.key} · {FIELD_TYPE_LABELS[field.type]}{field.required ? " · 必填" : ""}{field.semantic ? " · 语义字段" : ""}</span></div>
                  <button type="button" className="text-[10px] text-destructive hover:underline" onClick={() => setMetadataSchema((schema) => ({ ...schema, fields: schema.fields.filter((item) => item.key !== field.key) }))}>移除</button>
                </div>
              ))}
            </div>
          ) : null}
          <div className="space-y-2 rounded-md border bg-muted/20 p-2">
            <div className="grid grid-cols-2 gap-2"><Input value={fieldKey} onChange={(event) => setFieldKey(event.target.value)} placeholder="字段标识（英文小写）" className="h-8 text-xs" /><Input value={fieldLabel} onChange={(event) => setFieldLabel(event.target.value)} placeholder="显示名称" className="h-8 text-xs" /></div>
            <div className="grid grid-cols-2 gap-2"><select value={fieldType} onChange={(event) => setFieldType(event.target.value as MetadataFieldType)} className="h-8 rounded-md border bg-background px-2 text-xs"><option value="string">文本</option><option value="enum">单选</option><option value="multi_enum">多选</option><option value="date">日期</option><option value="datetime">日期时间</option><option value="integer">整数</option><option value="number">数值</option><option value="boolean">布尔值</option></select><Input value={fieldOptions} onChange={(event) => setFieldOptions(event.target.value)} disabled={fieldType !== "enum" && fieldType !== "multi_enum"} placeholder="选项，以英文逗号分隔" className="h-8 text-xs" /></div>
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px]"><label className="flex items-center gap-1"><input type="checkbox" checked={fieldRequired} onChange={(event) => setFieldRequired(event.target.checked)} />必填</label><label className="flex items-center gap-1"><input type="checkbox" checked={fieldFilterable} onChange={(event) => setFieldFilterable(event.target.checked)} />可筛选</label><label className="flex items-center gap-1"><input type="checkbox" checked={fieldSemantic} onChange={(event) => setFieldSemantic(event.target.checked)} />语义字段（修改后需重新建索引）</label></div>
            <Button type="button" variant="outline" size="sm" className="h-7 text-xs" onClick={addMetadataField}>添加元数据字段</Button>
          </div>
        </div>

      </div>

      <div className="flex flex-shrink-0 items-center justify-between border-t px-3 py-2">
        <Button variant="ghost" size="sm" onClick={() => setLlmMode("cloud")} className="h-7 gap-1 text-xs"><RotateCcw className="h-3 w-3" />恢复默认值</Button>
        <div className="flex items-center gap-1.5">
          <Button variant="ghost" size="sm" onClick={onClose} className="h-7 text-xs">取消</Button>
          <Button size="sm" onClick={handleSave} disabled={!hasChanges || saving} className="h-7 gap-1 text-xs"><Save className="h-3 w-3" />{saving ? "保存中..." : "保存"}</Button>
        </div>
      </div>
    </div>
  );
}
