import { QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { describe, expect, it } from "vitest";

import { useDeleteRating, useUpsertRating } from "@/features/ratings/hooks";
import { queryKeys } from "@/lib/query/keys";
import { dune, makeRating, pageOf, testUser } from "@/test/fixtures";
import { createTestQueryClient } from "@/test/test-utils";

function createWrapper() {
  const queryClient = createTestQueryClient();
  function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { queryClient, Wrapper };
}

function seedRatingCaches(queryClient: ReturnType<typeof createTestQueryClient>) {
  queryClient.setQueryData(queryKeys.ratings.item(dune.id), null);
  queryClient.setQueryData(queryKeys.ratings.list(0, 20), pageOf([]));
  queryClient.setQueryData(queryKeys.recommendations.mine(20), { is_fallback: true, items: [], limit: 20, source: "popularity" });
  queryClient.setQueryData(queryKeys.movies.detail(dune.id), dune);
  queryClient.setQueryData(queryKeys.movies.list({ limit: 20 }), pageOf([dune]));
  queryClient.setQueryData(queryKeys.user.me(), testUser);
}

describe("rating cache synchronization", () => {
  it("stores the upserted rating and invalidates dependent movie and list data", async () => {
    const { Wrapper, queryClient } = createWrapper();
    seedRatingCaches(queryClient);
    const { result } = renderHook(() => useUpsertRating(), { wrapper: Wrapper });

    await act(async () => {
      await result.current.mutateAsync({ movie_id: dune.id, score: 4.5 });
    });

    expect(queryClient.getQueryData(queryKeys.ratings.item(dune.id))).toMatchObject({ score: 4.5 });
    expect(queryClient.getQueryState(queryKeys.ratings.list(0, 20))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.recommendations.mine(20))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.movies.detail(dune.id))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.movies.list({ limit: 20 }))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.user.me())?.isInvalidated).toBe(false);
  });

  it("clears the item cache and invalidates dependents after deleting a rating", async () => {
    const { Wrapper, queryClient } = createWrapper();
    seedRatingCaches(queryClient);
    queryClient.setQueryData(queryKeys.ratings.item(dune.id), makeRating(dune, 3));
    const { result } = renderHook(() => useDeleteRating(), { wrapper: Wrapper });

    await act(async () => {
      await result.current.mutateAsync(dune.id);
    });

    expect(queryClient.getQueryData(queryKeys.ratings.item(dune.id))).toBeNull();
    expect(queryClient.getQueryState(queryKeys.ratings.list(0, 20))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.recommendations.mine(20))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.movies.detail(dune.id))?.isInvalidated).toBe(true);
  });
});
