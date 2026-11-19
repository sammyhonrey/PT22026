import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { deleteRating, fetchMyRating, fetchMyRatings, upsertRating } from "@/features/ratings/api";
import { isApiClientError } from "@/lib/api/errors";
import { queryKeys } from "@/lib/query/keys";

function invalidateAfterRating(queryClient: ReturnType<typeof useQueryClient>, movieId: string) {
  void queryClient.invalidateQueries({ queryKey: queryKeys.ratings.item(movieId) });
  void queryClient.invalidateQueries({ queryKey: ["ratings", "list"] });
  void queryClient.invalidateQueries({ queryKey: ["recommendations", "mine"] });
  void queryClient.invalidateQueries({ queryKey: queryKeys.movies.detail(movieId) });
  void queryClient.invalidateQueries({ queryKey: ["movies"] });
}

export function useMyRating(movieId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.ratings.item(movieId),
    queryFn: async ({ signal }) => {
      try {
        return await fetchMyRating(movieId, signal);
      } catch (error) {
        if (isApiClientError(error) && error.status === 404) return null;
        throw error;
      }
    },
    enabled,
    staleTime: 30_000,
  });
}

export function useMyRatings(offset = 0, limit = 20, enabled = true) {
  return useQuery({
    queryKey: queryKeys.ratings.list(offset, limit),
    queryFn: ({ signal }) => fetchMyRatings(offset, limit, signal),
    enabled,
    staleTime: 30_000,
  });
}

export function useUpsertRating() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: upsertRating,
    onSuccess: (rating) => {
      queryClient.setQueryData(queryKeys.ratings.item(rating.movie_id), rating);
      invalidateAfterRating(queryClient, rating.movie_id);
    },
  });
}

export function useDeleteRating() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteRating,
    onSuccess: (_, movieId) => {
      queryClient.setQueryData(queryKeys.ratings.item(movieId), null);
      invalidateAfterRating(queryClient, movieId);
    },
  });
}
