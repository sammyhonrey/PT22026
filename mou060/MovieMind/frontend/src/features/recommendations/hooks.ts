import { useQuery } from "@tanstack/react-query";

import { fetchMyRecommendations } from "@/features/recommendations/api";
import { queryKeys } from "@/lib/query/keys";

export function useMyRecommendations(limit = 20, enabled = true) {
  return useQuery({
    queryKey: queryKeys.recommendations.mine(limit),
    queryFn: ({ signal }) => fetchMyRecommendations(limit, signal),
    enabled,
    staleTime: 60_000,
  });
}
