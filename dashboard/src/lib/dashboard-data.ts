export type RangeKey = "1h" | "6h" | "24h";
export type Complexity = "simple" | "medium" | "complex";

export interface RouteEvent {
  id: number;
  time: string;
  complexity: Complexity;
  route: string;
  prompt: string;
  latency: number;
}

export const volumeByRange: Record<RangeKey, { label: string; requests: number }[]> = {
  "1h": [
    { label: "19:00", requests: 8 }, { label: "19:10", requests: 13 },
    { label: "19:20", requests: 9 }, { label: "19:30", requests: 18 },
    { label: "19:40", requests: 25 }, { label: "19:50", requests: 16 },
  ],
  "6h": [
    { label: "14:00", requests: 18 }, { label: "15:00", requests: 25 },
    { label: "16:00", requests: 42 }, { label: "17:00", requests: 36 },
    { label: "18:00", requests: 67 }, { label: "19:00", requests: 54 },
  ],
  "24h": [
    { label: "20:00", requests: 12 }, { label: "00:00", requests: 18 },
    { label: "04:00", requests: 15 }, { label: "08:00", requests: 32 },
    { label: "12:00", requests: 46 }, { label: "14:00", requests: 41 },
    { label: "16:00", requests: 74 }, { label: "17:00", requests: 112 },
    { label: "18:00", requests: 61 }, { label: "19:00", requests: 34 },
  ],
};

export const distribution = [
  { name: "fast-code", value: 142, color: "var(--chart-1)" },
  { name: "balanced", value: 108, color: "var(--chart-2)" },
  { name: "deep-reason", value: 65, color: "var(--chart-3)" },
  { name: "fallback", value: 19, color: "var(--chart-4)" },
];

export const destinationTraffic = [
  { name: "fast-code", requests: 142, fill: "var(--chart-1)" },
  { name: "balanced", requests: 108, fill: "var(--chart-2)" },
  { name: "deep-reason", requests: 65, fill: "var(--chart-3)" },
];

export const initialEvents: RouteEvent[] = [
  { id: 1, time: "19:42:18", complexity: "complex", route: "deep-reason", prompt: "Refactor an async Python task queue with retry policies", latency: 1842 },
  { id: 2, time: "19:41:52", complexity: "simple", route: "fast-code", prompt: "Write a typed utility to flatten nested dictionaries", latency: 462 },
  { id: 3, time: "19:41:09", complexity: "medium", route: "balanced", prompt: "Add pagination and caching to a FastAPI endpoint", latency: 927 },
  { id: 4, time: "19:40:44", complexity: "complex", route: "deep-reason", prompt: "Review a distributed worker architecture for race conditions", latency: 2136 },
  { id: 5, time: "19:39:21", complexity: "simple", route: "fast-code", prompt: "Explain this regular expression and add examples", latency: 388 },
  { id: 6, time: "19:38:57", complexity: "medium", route: "balanced", prompt: "Generate unit tests for a repository service", latency: 814 },
];

export const incomingEvents: Omit<RouteEvent, "id" | "time">[] = [
  { complexity: "simple", route: "fast-code", prompt: "Create a Python dataclass for runtime configuration", latency: 416 },
  { complexity: "medium", route: "balanced", prompt: "Optimize a streaming parser for large JSON responses", latency: 863 },
  { complexity: "complex", route: "deep-reason", prompt: "Design a fault-tolerant routing policy with circuit breakers", latency: 1924 },
];