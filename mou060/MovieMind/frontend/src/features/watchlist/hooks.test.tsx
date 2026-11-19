import { QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { describe, expect, it } from "vitest";

import { useAddToWatchlist, useRemoveFromWatchlist } from "@/features/watchlist/hooks";
import { queryKeys } from "@/lib/query/keys";
import { dune, makeWatchlistItem, pageOf } from "@/test/fixtures";
import { createTestQueryClient } from "@/test/test-utils";

function createWrapper() {
  const queryClient = createTestQueryClient();
  function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { queryClient, Wrapper };
}

function seedWatchlistCaches(queryClient: ReturnType<typeof createTestQueryClient>) {
  queryClient.setQueryData(queryKeys.watchlist.list(0, 100), pageOf([], 100));
  queryClient.setQueryData(queryKeys.recommendations.mine(20), { is_fallback: true, items: [], limit: 20, source: "popularity" });
  queryClient.setQueryData(queryKeys.movies.detail(dune.id), dune);
}

describe("watchlist cache synchronization", () => {
  it("invalidates saved-movie and recommendation data after adding", async () => {
    const { Wrapper, queryClient } = createWrapper();
    seedWatchlistCaches(queryClient);
    const { result } = renderHook(() => useAddToWatchlist(), { wrapper: Wrapper });

    await act(async () => {
      await result.current.mutateAsync(dune.id);
    });

    expect(queryClient.getQueryState(queryKeys.watchlist.list(0, 100))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.recommendations.mine(20))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.movies.detail(dune.id))?.isInvalidated).toBe(false);
  });

  it("invalidates saved-movie and recommendation data after removing", async () => {
    const { Wrapper, queryClient } = createWrapper();
    seedWatchlistCaches(queryClient);
    queryClient.setQueryData(queryKeys.watchlist.list(0, 100), pageOf([makeWatchlistItem(dune)], 100));
    const { result } = renderHook(() => useRemoveFromWatchlist(), { wrapper: Wrapper });

    await act(async () => {
      await result.current.mutateAsync(dune.id);
    });

    expect(queryClient.getQueryState(queryKeys.watchlist.list(0, 100))?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(queryKeys.recommendations.mine(20))?.isInvalidated).toBe(true);
  });
});
