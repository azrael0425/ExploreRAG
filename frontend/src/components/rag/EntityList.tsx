import { useState, useMemo, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Search,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Users,
  Building2,
  MapPin,
  Lightbulb,
  Calendar,
  Tag,
  Loader2,
  Network,
  Link2,
  SlidersHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getEntityTypeColor, getEntityTypeLabel, normalizeEntityType } from "@/lib/entityTypes";
import { api } from "@/lib/api";
import type { KGEntity, KGRelationship } from "@/types";

// ---------------------------------------------------------------------------
// Entity type config — icon + color
// ---------------------------------------------------------------------------
const ENTITY_TYPE_ICONS: Record<string, typeof Tag> = {
  person: Users,
  organization: Building2,
  organisation: Building2,
  company: Building2,
  institution: Building2,
  location: MapPin,
  country: MapPin,
  event: Calendar,
  date: Calendar,
  concept: Lightbulb,
  technology: Lightbulb,
};

function getEntityConfig(type: string) {
  const key = normalizeEntityType(type);
  return {
    icon: ENTITY_TYPE_ICONS[key] ?? Tag,
    color: getEntityTypeColor(type),
  };
}

// ---------------------------------------------------------------------------
// TypeBadge
// ---------------------------------------------------------------------------
function TypeBadge({ type }: { type: string }) {
  const config = getEntityConfig(type);
  const Icon = config.icon;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium"
      style={{
        color: config.color,
        borderColor: `color-mix(in srgb, ${config.color} 35%, transparent)`,
        backgroundColor: `color-mix(in srgb, ${config.color} 10%, transparent)`,
      }}
    >
      <Icon className="w-3 h-3" />
      {getEntityTypeLabel(type)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// RelationshipRow — shown when entity is expanded
// ---------------------------------------------------------------------------
function RelationshipRow({ rel, entityName }: { rel: KGRelationship; entityName: string }) {
  const isSource = rel.source.toLowerCase() === entityName.toLowerCase();
  const other = isSource ? rel.target : rel.source;
  const source = isSource ? entityName : other;
  const target = isSource ? other : entityName;
  const detail = rel.description || rel.keywords || "相关";

  return (
    <div className="rounded-lg border bg-background/75 p-2.5 shadow-sm">
      <div className="flex min-w-0 items-center gap-2 text-xs">
        <span className="min-w-0 flex-1 break-words font-semibold text-foreground">{source}</span>
        <ArrowRight className="h-3.5 w-3.5 flex-shrink-0 text-primary/55" />
        <span className="min-w-0 flex-1 break-words text-right font-semibold text-foreground">{target}</span>
      </div>
      <p className="mt-2 whitespace-normal break-words text-[11px] leading-relaxed text-muted-foreground">
        {detail}
      </p>
      {(rel.keywords || rel.weight > 0) && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[9px] text-muted-foreground/70">
          {rel.keywords && rel.keywords !== detail && (
            <span className="rounded bg-muted px-1.5 py-0.5 break-all">{rel.keywords}</span>
          )}
          {rel.weight > 0 && <span>关联强度 {rel.weight.toFixed(2)}</span>}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EntityRow — single row in the entity table
// ---------------------------------------------------------------------------
const EntityRow = memo(function EntityRow({
  entity,
  projectId,
}: {
  entity: KGEntity;
  projectId: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);
  const entityConfig = getEntityConfig(entity.entity_type);
  const EntityIcon = entityConfig.icon;

  // Lazy-load relationships only when expanded
  const { data: relationships, isLoading: relsLoading } = useQuery({
    queryKey: ["kg-relationships", projectId, entity.name],
    queryFn: () => api.get<KGRelationship[]>(`/rag/relationships/${projectId}?entity=${encodeURIComponent(entity.name)}&limit=500`),
    enabled: expanded,
    staleTime: 60_000,
  });

  return (
    <div className={cn("border-b last:border-b-0 transition-colors", expanded && "bg-muted/20")}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="group w-full flex items-center gap-3 px-3 py-3 hover:bg-muted/45 transition-colors text-left"
        data-entity-name={entity.name}
      >
        <span
          className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border transition-transform group-hover:scale-105"
          style={{
            color: entityConfig.color,
            borderColor: `color-mix(in srgb, ${entityConfig.color} 28%, transparent)`,
            backgroundColor: `color-mix(in srgb, ${entityConfig.color} 9%, transparent)`,
          }}
        >
          <EntityIcon className="h-4 w-4" />
        </span>

        {/* Name */}
        <span className="min-w-0 flex-1">
          <span className="block break-words text-sm font-semibold leading-snug">{entity.name}</span>
          <span className="mt-0.5 block text-[10px] text-muted-foreground">{entity.degree} 个直接连接</span>
        </span>

        {/* Type badge */}
        <TypeBadge type={entity.entity_type} />

        {/* Degree */}
        {/* Expand */}
        {entity.degree > 0 && (
          expanded
            ? <ChevronUp className="w-4 h-4 text-primary flex-shrink-0" />
            : <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        )}
      </button>

      {/* Expanded: description + relationships */}
      {expanded && (
          <div className="px-3 pb-3">
            <div
              className="space-y-3 rounded-xl border-l-2 bg-background/70 p-3"
              style={{ borderLeftColor: entityConfig.color }}
            >
              {/* Description */}
              {entity.description && (
                <div className="rounded-lg border bg-muted/20 p-2.5">
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-semibold text-muted-foreground">实体说明</span>
                    <button
                      type="button"
                      onClick={() => setDescriptionExpanded((value) => !value)}
                      className="flex-shrink-0 text-[10px] font-medium text-primary hover:underline"
                    >
                      {descriptionExpanded ? "收起说明" : "展开完整说明"}
                    </button>
                  </div>
                  <p className={cn(
                    "whitespace-normal break-words text-xs leading-relaxed text-muted-foreground/85",
                    !descriptionExpanded && "line-clamp-4"
                  )}>
                    {entity.description}
                  </p>
                </div>
              )}

              {/* Relationships */}
              {relsLoading && (
                <div className="flex items-center gap-2 py-2 px-1">
                  <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">正在加载关系...</span>
                </div>
              )}
              {relationships && relationships.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[10px] font-medium text-muted-foreground">
                    <span className="flex items-center gap-1.5">
                      <Link2 className="h-3 w-3" />
                      关联实体
                    </span>
                    <span>{relationships.length} 条关系</span>
                  </div>
                  {relationships.map((rel, i) => (
                    <RelationshipRow key={i} rel={rel} entityName={entity.name} />
                  ))}
                </div>
              )}
              {relationships && relationships.length === 0 && (
                <p className="text-xs text-muted-foreground/50">未找到关系</p>
              )}
            </div>
          </div>
        )}
    </div>
  );
});

// ---------------------------------------------------------------------------
// Sort options
// ---------------------------------------------------------------------------
type SortKey = "degree" | "name" | "type";

// ---------------------------------------------------------------------------
// EntityList
// ---------------------------------------------------------------------------
interface EntityListProps {
  projectId: string;
  highlightEntities?: string[];
}

export const EntityList = memo(function EntityList({ projectId, highlightEntities = [] }: EntityListProps) {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortKey>("degree");

  const { data: entities, isLoading } = useQuery({
    queryKey: ["kg-entities", projectId],
    queryFn: () => api.get<KGEntity[]>(`/rag/entities/${projectId}?limit=500`),
    staleTime: 30_000,
  });

  // Unique entity types for filter
  const entityTypes = useMemo(() => {
    if (!entities) return [];
    const types = new Set(entities.map((e) => e.entity_type));
    return Array.from(types).sort();
  }, [entities]);

  // Filter + sort
  const filtered = useMemo(() => {
    if (!entities) return [];
    let result = entities;

    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter((e) => e.name.toLowerCase().includes(q) || e.description.toLowerCase().includes(q));
    }

    if (typeFilter) {
      result = result.filter((e) => e.entity_type.toLowerCase() === typeFilter.toLowerCase());
    }

    if (sortBy === "name") {
      result = [...result].sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortBy === "type") {
      result = [...result].sort((a, b) => a.entity_type.localeCompare(b.entity_type) || b.degree - a.degree);
    }
    // Default "degree" sort is already from API

    return result;
  }, [entities, search, typeFilter, sortBy]);

  if (isLoading) {
    return (
      <div className="h-full space-y-2">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-10 rounded-md bg-muted" />
        ))}
      </div>
    );
  }

  if (!entities || entities.length === 0) {
    return (
      <div className="flex flex-col items-center py-10 text-center">
        <Network className="w-10 h-10 text-muted-foreground/30 mb-3" />
        <p className="text-sm text-muted-foreground">尚未提取到实体</p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          处理文档时，系统会自动提取实体
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* Controls: search + filters */}
      <div className="flex-shrink-0 rounded-xl border bg-card/40 p-2.5 shadow-sm">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold">
            <SlidersHorizontal className="h-3.5 w-3.5 text-primary" />
            实体浏览器
          </div>
          <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
            {filtered.length} / {entities.length}
          </span>
        </div>

        <div className="flex gap-2 flex-wrap">
        {/* Search */}
        <div className="relative flex-1 min-w-[180px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
          <input
            type="text"
            placeholder="搜索实体..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full h-8 pl-8 pr-3 rounded-lg border border-input bg-background text-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/15"
          />
        </div>

        {/* Type filter */}
        <select
          value={typeFilter ?? ""}
          onChange={(e) => setTypeFilter(e.target.value || null)}
          className="h-8 px-2 rounded-lg border border-input bg-background text-xs"
        >
          <option value="">全部类型</option>
          {entityTypes.map((t) => (
            <option key={t} value={t}>{getEntityTypeLabel(t)}</option>
          ))}
        </select>

        {/* Sort */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortKey)}
          className="h-8 px-2 rounded-lg border border-input bg-background text-xs"
        >
          <option value="degree">连接数最多</option>
          <option value="name">按名称排序</option>
          <option value="type">按类型排序</option>
        </select>
        </div>

        {entityTypes.length > 1 && (
          <div className="mt-2 flex gap-1.5 overflow-x-auto pb-0.5">
            <button
              type="button"
              onClick={() => setTypeFilter(null)}
              className={cn(
                "flex-shrink-0 rounded-full border px-2 py-0.5 text-[10px] transition-colors",
                typeFilter === null
                  ? "border-primary/30 bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted"
              )}
            >
              全部
            </button>
            {entityTypes.map((type) => {
              const color = getEntityTypeColor(type);
              const active = typeFilter?.toLowerCase() === type.toLowerCase();
              return (
                <button
                  key={type}
                  type="button"
                  onClick={() => setTypeFilter(active ? null : type)}
                  className="flex flex-shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] transition-opacity hover:opacity-80"
                  style={{
                    color,
                    borderColor: `color-mix(in srgb, ${color} ${active ? 55 : 25}%, transparent)`,
                    backgroundColor: `color-mix(in srgb, ${color} ${active ? 13 : 5}%, transparent)`,
                  }}
                >
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                  {getEntityTypeLabel(type)}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Count + type chips */}
      <div className="flex flex-shrink-0 items-center gap-2 flex-wrap px-1">
        <span className="text-xs text-muted-foreground">
          {filtered.length} 个实体
        </span>
        {typeFilter && (
          <button
            onClick={() => setTypeFilter(null)}
            className="text-[10px] text-primary bg-primary/10 px-2 py-0.5 rounded-full hover:bg-primary/20 transition-colors"
          >
            {getEntityTypeLabel(typeFilter)} &times;
          </button>
        )}
      </div>

      {/* Entity rows */}
      <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border bg-card/25 shadow-sm">
        <div className="sticky top-0 z-10 flex items-center gap-3 border-b bg-background/95 px-3 py-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground backdrop-blur-sm">
          <span className="flex-1">实体与连接</span>
          <span>类型</span>
          <span className="w-4" />
        </div>
        {filtered.map((entity) => {
          const isHighlighted = highlightEntities.some(
            (e) => e.toLowerCase() === entity.name.toLowerCase()
          );
          return (
            <div
              key={entity.name}
              className={cn(
                isHighlighted && "bg-amber-400/10 border-l-2 border-l-amber-400"
              )}
            >
              <EntityRow entity={entity} projectId={projectId} />
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && entities.length > 0 && (
        <p className="text-center text-xs text-muted-foreground py-4">
          没有符合筛选条件的实体
        </p>
      )}
    </div>
  );
});
