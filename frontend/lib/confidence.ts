export const CONFIDENCE_THRESHOLDS = {
  high: 0.8,
  medium: 0.6,
  low: 0.4,
} as const;

export const CONFIDENCE_BADGE_COLORS = {
  high: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
  medium: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300",
  low: "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300",
  critical: "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300",
} as const;

export const CONFIDENCE_DOT_COLORS = {
  high: "bg-green-500",
  medium: "bg-yellow-500",
  low: "bg-orange-500",
  critical: "bg-red-500",
} as const;

export interface ConfidenceTier {
  tier: "high" | "medium" | "low" | "critical";
  label: string;
  badgeColor: string;
  dotColor: string;
}

export function getConfidenceTier(conf: number): ConfidenceTier {
  if (conf >= CONFIDENCE_THRESHOLDS.high) {
    return { tier: "high", label: "高", badgeColor: CONFIDENCE_BADGE_COLORS.high, dotColor: CONFIDENCE_DOT_COLORS.high };
  }
  if (conf >= CONFIDENCE_THRESHOLDS.medium) {
    return { tier: "medium", label: "中", badgeColor: CONFIDENCE_BADGE_COLORS.medium, dotColor: CONFIDENCE_DOT_COLORS.medium };
  }
  if (conf >= CONFIDENCE_THRESHOLDS.low) {
    return { tier: "low", label: "低", badgeColor: CONFIDENCE_BADGE_COLORS.low, dotColor: CONFIDENCE_DOT_COLORS.low };
  }
  return { tier: "critical", label: "极低", badgeColor: CONFIDENCE_BADGE_COLORS.critical, dotColor: CONFIDENCE_DOT_COLORS.critical };
}

export function getConfidenceBadgeColor(conf: number): string {
  return getConfidenceTier(conf).badgeColor;
}

export function getConfidenceDotColor(conf: number): string {
  return getConfidenceTier(conf).dotColor;
}

export function isLowConfidence(conf: number): boolean {
  return conf < CONFIDENCE_THRESHOLDS.low;
}