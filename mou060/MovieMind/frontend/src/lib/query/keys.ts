import type { QueryKey } from "@tanstack/react-query";

export type MovieListQuery = {
  direction?: "asc" | "desc";
  genre?: string;
  limit?: number;
  offset?: number;
  q?: string;
  sort?: "average_rating" | "created_at" | "ratings_count" | "release_year" | "title";
  year?: number;
};

export const queryKeys = {
  movies: {
    detail: (movieId: string) => ["movies", "detail", movieId] as const,
    genres: () => ["movies", "genres"] as const,
    list: (query: MovieListQuery) => ["movies", "list", query] as const,
    popular: (limit = 20) => ["movies", "popular", { limit }] as const,
  },
  ratings: {
    item: (movieId: string) => ["ratings", "item", movieId] as const,
    list: (offset = 0, limit = 20) => ["ratings", "list", { limit, offset }] as const,
  },
  preferences: {
    mine: () => ["preferences", "mine"] as const,
  },
  recommendations: {
    mine: (limit = 20) => ["recommendations", "mine", { limit }] as const,
  },
  user: {
    me: () => ["user", "me"] as const,
  },
  watchlist: {
    list: (offset = 0, limit = 20) => ["watchlist", "list", { limit, offset }] as const,
  },
};

const privateRoots = new Set(["ratings", "preferences", "recommendations", "user", "watchlist"]);

export function isUserSpecificQuery(queryKey: QueryKey) {
  return typeof queryKey[0] === "string" && privateRoots.has(queryKey[0]);
}
