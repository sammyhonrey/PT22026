import { apiClient } from "@/lib/api/client";
import type { Rating, RatingList } from "@/types/api";

export async function fetchMyRating(movieId: string, signal?: AbortSignal) {
  const { data } = await apiClient.get<Rating>(`/ratings/me/${movieId}`, { signal });
  return data;
}

export async function fetchMyRatings(offset = 0, limit = 20, signal?: AbortSignal) {
  const { data } = await apiClient.get<RatingList>("/ratings/me", { params: { limit, offset }, signal });
  return data;
}

export async function upsertRating(payload: { movie_id: string; score: number }) {
  const { data } = await apiClient.post<Rating>("/ratings", payload);
  return data;
}

export async function deleteRating(movieId: string) {
  await apiClient.delete(`/ratings/me/${movieId}`);
}
