import { apiClient } from "@/lib/api/client";
import type { Watchlist, WatchlistItem } from "@/types/api";

export async function fetchWatchlist(offset = 0, limit = 20, signal?: AbortSignal) {
  const { data } = await apiClient.get<Watchlist>("/watchlist", { params: { limit, offset }, signal });
  return data;
}

export async function addToWatchlist(movieId: string) {
  const { data } = await apiClient.post<WatchlistItem>(`/watchlist/${movieId}`);
  return data;
}

export async function removeFromWatchlist(movieId: string) {
  await apiClient.delete(`/watchlist/${movieId}`);
}
