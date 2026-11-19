import { apiClient } from "@/lib/api/client";
import type { Recommendations, UserPreference } from "@/types/api";

export type PreferenceInput = {
  favorite_genres: string[];
  favorite_movie_ids: string[];
  description?: string | null;
};

export async function fetchPreferences(signal?: AbortSignal) {
  const { data } = await apiClient.get<UserPreference | null>("/preferences/me", { signal });
  return data;
}

export async function savePreferences(payload: PreferenceInput, signal?: AbortSignal) {
  const { data } = await apiClient.put<UserPreference>("/preferences/me", payload, { signal });
  return data;
}

export async function generatePreferenceRecommendations(payload: PreferenceInput, signal?: AbortSignal) {
  const { data } = await apiClient.post<Recommendations>("/preferences/me/recommend", payload, { signal });
  return data;
}
