import { describe, expect, it } from "vitest";

import {
  getEntityTypeColor,
  getEntityTypeLabel,
  normalizeEntityType,
} from "./entityTypes";

describe("entity type presentation", () => {
  it("normalizes backend identifiers consistently", () => {
    expect(normalizeEntityType(" Financial-Metric ")).toBe("financial_metric");
  });

  it("uses localized labels for known types and preserves unknown labels", () => {
    expect(getEntityTypeLabel("Organization")).toBe("组织");
    expect(getEntityTypeLabel("Custom Type")).toBe("Custom Type");
  });

  it("returns stable visible colors for custom types", () => {
    const first = getEntityTypeColor("Custom Type");
    expect(first).toBe(getEntityTypeColor("custom-type"));
    expect(first).toMatch(/^hsl\(\d+ 68% 52%\)$/);
  });
});
