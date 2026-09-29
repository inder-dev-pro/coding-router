/**
 * React-Query hooks for every router dashboard API endpoint.
 *
 * Each hook polls at the given interval and returns `null` when the
 * Python backend is unreachable, which the UI maps to the "not connected" state.
 */

import { useQuery } from "@tanstack/react-query";
import {
  fetchSummary,
  fetchModels,
  fetchProviders,
  fetchCategories,
  fetchVolume,
  fetchEvents,
  fetchCost,
  fetchModes,
  fetchTiers,
  fetchEventDetail,
} from "@/lib/api";

const POLL_MS = 5_000; // refresh every 5 seconds

export function useSummary() {
  return useQuery({
    queryKey: ["router", "summary"],
    queryFn: fetchSummary,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["router", "models"],
    queryFn: fetchModels,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useProviders() {
  return useQuery({
    queryKey: ["router", "providers"],
    queryFn: fetchProviders,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useCategories() {
  return useQuery({
    queryKey: ["router", "categories"],
    queryFn: fetchCategories,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useVolume() {
  return useQuery({
    queryKey: ["router", "volume"],
    queryFn: fetchVolume,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useEvents() {
  return useQuery({
    queryKey: ["router", "events"],
    queryFn: fetchEvents,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useCostOverTime() {
  return useQuery({
    queryKey: ["router", "cost"],
    queryFn: fetchCost,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useModes() {
  return useQuery({
    queryKey: ["router", "modes"],
    queryFn: fetchModes,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useTiers() {
  return useQuery({
    queryKey: ["router", "tiers"],
    queryFn: fetchTiers,
    refetchInterval: POLL_MS,
    staleTime: POLL_MS - 500,
  });
}

export function useEventDetail(id: number | null) {
  return useQuery({
    queryKey: ["router", "event", id],
    queryFn: () => (id !== null ? fetchEventDetail(id) : Promise.resolve(null)),
    enabled: id !== null,
    staleTime: 30_000,
  });
}
