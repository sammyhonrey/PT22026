import { apiClient } from "@/lib/api/client";
import type { Recommendations } from "@/types/api";

export async function fetchMyRecommendations(limit = 20, signal?: AbortSignal) {
  const { data } = await apiClient.get<Recommendations>("/recommendations/me", { params: { limit }, signal });
  return data;
}
