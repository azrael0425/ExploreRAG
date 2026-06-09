/**
 * Confirm Dialog Component
 * ========================
 *
 * A styled confirmation dialog that replaces window.confirm().
 * Supports customizable title, message, and button labels.
 */

import { memo, useCallback, useEffect, useRef } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "./button";
import { cn } from "@/lib/utils";

interface ConfirmDialogProps {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "default";
}

export const ConfirmDialog = memo(function ConfirmDialog({
  open,
  onConfirm,
  onCancel,
  title = "确认操作",
  message,
  confirmLabel = "确认",
  cancelLabel = "取消",
  variant = "default",
}: ConfirmDialogProps) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  // Focus cancel button when opened
  useEffect(() => {
    if (open) {
      cancelRef.current?.focus();
    }
  }, [open]);

  // Close on Escape
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    },
    [onCancel]
  );

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onKeyDown={handleKeyDown}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
        onClick={onCancel}
      />

      {/* Dialog */}
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        aria-describedby="confirm-message"
        className="relative z-10 w-full max-w-md mx-4 rounded-xl bg-card border border-border shadow-2xl animate-in zoom-in-95 fade-in duration-200"
      >
        <div className="p-6">
          {/* Icon + Title */}
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center",
                variant === "danger"
                  ? "bg-destructive/15"
                  : "bg-primary/15"
              )}
            >
              <AlertTriangle
                className={cn(
                  "w-5 h-5",
                  variant === "danger" ? "text-destructive" : "text-primary"
                )}
              />
            </div>
            <div className="flex-1 min-w-0">
              <h3
                id="confirm-title"
                className="text-lg font-semibold leading-tight"
              >
                {title}
              </h3>
              <p
                id="confirm-message"
                className="mt-2 text-sm text-muted-foreground leading-relaxed"
              >
                {message}
              </p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-border/50 bg-muted/20 rounded-b-xl">
          <Button
            ref={cancelRef}
            variant="ghost"
            size="sm"
            onClick={onCancel}
          >
            {cancelLabel}
          </Button>
          <Button
            variant={variant === "danger" ? "destructive" : "default"}
            size="sm"
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
});
