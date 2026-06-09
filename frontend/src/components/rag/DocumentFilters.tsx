import { memo } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "@/types";

type FilterStatus = "all" | DocumentStatus;

const TABS: { value: FilterStatus; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "indexed", label: "就绪" },
  { value: "parsing", label: "处理中" },
  { value: "failed", label: "失败" },
];

interface DocumentFiltersProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  statusFilter: FilterStatus;
  onStatusChange: (s: FilterStatus) => void;
  counts: Record<FilterStatus, number>;
}

export type { FilterStatus };

export const DocumentFilters = memo(function DocumentFilters({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusChange,
  counts,
}: DocumentFiltersProps) {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      {/* Search */}
      <div className="relative flex-1 min-w-[180px] max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        <Input
          placeholder="按名称筛选..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-8 h-8 text-sm"
        />
      </div>

      {/* Status tabs */}
      <div className="flex items-center gap-1 bg-muted/40 rounded-lg p-0.5">
        {TABS.map((tab) => {
          const isActive = statusFilter === tab.value;
          // Merge processing-like statuses into the "Processing" tab
          let count = counts[tab.value] ?? 0;
          if (tab.value === "parsing") {
            count = (counts.parsing ?? 0) + (counts.indexing ?? 0) + (counts.processing ?? 0);
          }
          return (
            <button
              key={tab.value}
              onClick={() => onStatusChange(tab.value)}
              className={cn(
                "px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
                isActive
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tab.label}
              {count > 0 && (
                <span className={cn(
                  "ml-1 text-[10px]",
                  isActive ? "text-primary" : "text-muted-foreground/60"
                )}>
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
});
