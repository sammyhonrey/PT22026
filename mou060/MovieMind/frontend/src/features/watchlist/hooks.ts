import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { addToWatchlist, fetchWatchlist, removeFromWatchlist } from "@/features/watchlist/api";
import { queryKeys } from "@/lib/query/keys";

function invalidateAfterWatchlistMutation(queryClient: ReturnType<typeof useQueryClient>) {
  void queryClient.invalidateQueries({ queryKey: ["watchlist", "list"] });
  void queryClient.invalidateQueries({ queryKey: ["recommendations", "mine"] });
}

export function useWatchlist(offset = 0, limit = 20, enabled = true) {
  return useQuery({
    queryKey: queryKeys.watchlist.list(offset, limit),
    queryFn: ({ signal }) => fetchWatchlist(offset, limit, signal),
    enabled,
    staleTime: 30_000,
  });
}

export function useAddToWatchlist() {
  const queryClient = useQueryClient();
  return useMutation({ mutationFn: addToWatchlist, onSuccess: () => invalidateAfterWatchlistMutation(queryClient) });
}

export function useRemoveFromWatchlist() {
  const queryClient = useQueryClient();
  return useMutation({ mutationFn: removeFromWatchlist, onSuccess: () => invalidateAfterWatchlistMutation(queryClient) });
}
