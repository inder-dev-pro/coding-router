/**
 * API client for the coding-router Python backend.
 *
 * The backend runs at http://127.0.0.1:3000 and exposes a JSON API.
 * All functions return `null` when the backend is unreachable so the
 * dashboard can show a graceful "router not connected" state.
 */

const API_BASE = (typeof window !== "undefined" && window.location.port !== "8080" && window.location.port !== "8081") ? "" : "http://127.0.0.1:3000";

// ---------------------------------------------------------------------------
// Response types — mirror the Python tracker's query helpers
// ---------------------------------------------------------------------------

export interface SummaryStats {
  total_requests: number;
  providers: number;
  models_active: number;
  total_spend: number;
  avg_latency: number;
  error_count: number;
  requests_today: number;
}

export interface ModelDistribution {
  model: string;
  service: string;
  count: number;
}

export interface ProviderDistribution {
  provider: string;
  count: number;
}

export interface CategoryDistribution {
  category: string;
  count: number;
}

export interface VolumePoint {
  hour_bucket: number;
  count: number;
  bucket_start: number;
}

export interface RecentEvent {
  id: number;
  timestamp: number;
  query_preview: string;
  category: string;
  routing_mode: string;
  selected_key: string;
  selected_service: string;
  selected_tier: string;
  similarity: number | null;
  reward: number | null;
  estimated_cost_usd: number | null;
  latency_ms: number | null;
  had_error: number;
  error_text: string | null;
}

export interface CostPoint {
  timestamp: number;
  estimated_cost_usd: number | null;
  cumulative_cost: number | null;
}

export interface ModeDistribution {
  mode: string;
  count: number;
}

export interface TierDistribution {
  tier: string;
  count: number;
}

export interface EventDetail {
  id: number;
  timestamp: number;
  query: string;
  query_preview: string;
  category: string;
  classifier_model: string | null;
  routing_mode: string;
  alpha: number | null;
  beta: number | null;
  selection_group: string;
  selection_reason: string | null;
  selected_key: string;
  selected_model: string;
  selected_service: string;
  selected_tier: string;
  similarity: number | null;
  reward: number | null;
  estimated_cost_usd: number | null;
  input_price: number | null;
  output_price: number | null;
  estimated_input_tokens: number | null;
  estimated_output_tokens: number | null;
  latency_ms: number | null;
  had_error: number;
  error_text: string | null;
  ranked_models_json: string | null;
}

// ---------------------------------------------------------------------------
// Generic fetch wrapper
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Endpoint-specific fetchers
// ---------------------------------------------------------------------------

export const fetchSummary = () => apiFetch<SummaryStats>("/api/summary");
export const fetchModels = () => apiFetch<ModelDistribution[]>("/api/models");
export const fetchProviders = () => apiFetch<ProviderDistribution[]>("/api/providers");
export const fetchCategories = () => apiFetch<CategoryDistribution[]>("/api/categories");
export const fetchVolume = () => apiFetch<VolumePoint[]>("/api/volume");
export const fetchEvents = () => apiFetch<RecentEvent[]>("/api/events");
export const fetchCost = () => apiFetch<CostPoint[]>("/api/cost");
export const fetchModes = () => apiFetch<ModeDistribution[]>("/api/modes");
export const fetchTiers = () => apiFetch<TierDistribution[]>("/api/tiers");
export const fetchEventDetail = (id: number) => apiFetch<EventDetail>(`/api/event/${id}`);
