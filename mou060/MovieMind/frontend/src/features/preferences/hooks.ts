import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { fetchPreferences, generatePreferenceRecommendations, savePreferences, type PreferenceInput } from "@/features/preferences/api";
import { queryKeys } from "@/lib/query/keys";

export function useMyPreferences(enabled = true) {
  return useQuery({
    queryKey: queryKeys.preferences.mine(),
    queryFn: ({ signal }) => fetchPreferences(signal),
    enabled,
    staleTime: 60_000,
  });
}

export function useSavePreferences() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: PreferenceInput) => savePreferences(payload),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.preferences.mine(), data);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", "mine"] });
    },
  });
}

export function useGeneratePreferenceRecommendations() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: PreferenceInput) => generatePreferenceRecommendations(payload),
    onSuccess: (data) => {
      const limit = data.limit || 20;
      queryClient.setQueryData(queryKeys.recommendations.mine(limit), data);
      void queryClient.invalidateQueries({ queryKey: ["recommendations", "mine"] });
    },
  });
}
