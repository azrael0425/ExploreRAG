import { useEffect, useState } from "react";
import { FileText, Save, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { MetadataFieldDefinition, MetadataSchema } from "@/types";

function MetadataFields({
  fields,
  values,
  onChange,
}: {
  fields: MetadataFieldDefinition[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  return (
    <div className="space-y-3">
      {fields.map((field) => {
        const value = values[field.key];
        const label = <span>{field.label}{field.required ? <span className="ml-0.5 text-destructive">*</span> : null}</span>;
        if (field.type === "boolean") {
          return <label key={field.key} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm"><span>{label}</span><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(field.key, event.target.checked)} /></label>;
        }
        if (field.type === "enum") {
          return <label key={field.key} className="block space-y-1 text-xs font-medium">{label}<select value={typeof value === "string" ? value : ""} onChange={(event) => onChange(field.key, event.target.value || undefined)} className="h-9 w-full rounded-md border bg-background px-2 text-sm font-normal"><option value="">请选择</option>{field.options.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>;
        }
        if (field.type === "multi_enum") {
          const selected = Array.isArray(value) ? value.map(String) : [];
          return <fieldset key={field.key} className="space-y-1"><legend className="text-xs font-medium">{label}</legend><div className="flex flex-wrap gap-1.5">{field.options.map((option) => <label key={option} className="inline-flex items-center gap-1 rounded border px-2 py-1 text-xs"><input type="checkbox" checked={selected.includes(option)} onChange={(event) => onChange(field.key, event.target.checked ? [...selected, option] : selected.filter((item) => item !== option))} />{option}</label>)}</div></fieldset>;
        }
        const inputType = field.type === "integer" || field.type === "number" ? "number" : field.type === "date" ? "date" : field.type === "datetime" ? "datetime-local" : "text";
        return <label key={field.key} className="block space-y-1 text-xs font-medium">{label}<Input type={inputType} value={typeof value === "string" || typeof value === "number" ? String(value) : ""} onChange={(event) => { const next = event.target.value; onChange(field.key, next === "" ? undefined : field.type === "integer" || field.type === "number" ? Number(next) : next); }} className="h-9 text-sm font-normal" /></label>;
      })}
    </div>
  );
}

interface DocumentMetadataDialogProps {
  open: boolean;
  title: string;
  schema: MetadataSchema;
  initialValues?: Record<string, unknown>;
  files?: File[];
  saving?: boolean;
  onClose: () => void;
  onSave: (metadata: Record<string, unknown>) => Promise<void>;
}

export function DocumentMetadataDialog({ open, title, schema, initialValues, files = [], saving, onClose, onSave }: DocumentMetadataDialogProps) {
  const [values, setValues] = useState<Record<string, unknown>>({});

  useEffect(() => {
    const defaults = Object.fromEntries(schema.fields.filter((field) => field.default !== undefined).map((field) => [field.key, field.default]));
    setValues({ ...defaults, ...(initialValues ?? {}) });
  }, [schema, initialValues, open]);

  if (!open) return null;
  const visibleValues = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined && value !== ""));

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 p-4">
      <div className="max-h-[85vh] w-full max-w-md overflow-hidden rounded-lg border bg-background shadow-xl">
        <div className="flex items-start justify-between border-b px-4 py-3"><div><h2 className="text-sm font-semibold">{title}</h2>{files.length > 0 ? <p className="mt-1 text-xs text-muted-foreground"><FileText className="mr-1 inline h-3 w-3" />将为 {files.length} 个文件应用相同元数据</p> : null}</div><Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}><X className="h-4 w-4" /></Button></div>
        <div className="max-h-[60vh] overflow-y-auto px-4 py-4">{schema.fields.length ? <MetadataFields fields={schema.fields} values={values} onChange={(key, value) => setValues((current) => ({ ...current, [key]: value }))} /> : <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">此知识库尚未配置导入元数据字段。可直接继续上传，或在知识库设置中添加字段。</p>}</div>
        <div className="flex justify-end gap-2 border-t px-4 py-3"><Button variant="ghost" size="sm" onClick={onClose}>取消</Button><Button size="sm" disabled={saving} onClick={() => void onSave(visibleValues)}><Save className="mr-1 h-3.5 w-3.5" />{saving ? "保存中..." : "确认"}</Button></div>
      </div>
    </div>
  );
}
