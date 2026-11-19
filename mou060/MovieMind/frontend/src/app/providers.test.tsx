import { act, render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createQueryClient, QueryProvider } from "@/app/providers";
import { authSession } from "@/lib/auth/session";
import { queryKeys } from "@/lib/query/keys";
import { dune, pageOf, testUser } from "@/test/fixtures";

describe("QueryProvider session cleanup", () => {
  it("removes protected cached data on sign-out while keeping public data", () => {
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.user.me(), testUser);
    queryClient.setQueryData(queryKeys.ratings.list(0, 20), pageOf([]));
    queryClient.setQueryData(queryKeys.watchlist.list(0, 20), pageOf([]));
    queryClient.setQueryData(queryKeys.recommendations.mine(20), { is_fallback: true, items: [], limit: 20, source: "popularity" });
    queryClient.setQueryData(queryKeys.movies.detail(dune.id), dune);
    queryClient.setQueryData(queryKeys.movies.list({ limit: 20 }), pageOf([dune]));
    authSession.setToken("test-token");

    render(
      <QueryProvider client={queryClient}>
        <div>child</div>
      </QueryProvider>,
    );

    act(() => authSession.clear());

    expect(queryClient.getQueryData(queryKeys.user.me())).toBeUndefined();
    expect(queryClient.getQueryData(queryKeys.ratings.list(0, 20))).toBeUndefined();
    expect(queryClient.getQueryData(queryKeys.watchlist.list(0, 20))).toBeUndefined();
    expect(queryClient.getQueryData(queryKeys.recommendations.mine(20))).toBeUndefined();
    expect(queryClient.getQueryData(queryKeys.movies.detail(dune.id))).toEqual(dune);
    expect(queryClient.getQueryData(queryKeys.movies.list({ limit: 20 }))).toEqual(pageOf([dune]));
  });

  it("keeps cached data when a token is issued", () => {
    const queryClient = createQueryClient();
    queryClient.setQueryData(queryKeys.user.me(), testUser);

    render(
      <QueryProvider client={queryClient}>
        <div>child</div>
      </QueryProvider>,
    );

    act(() => authSession.setToken("test-token"));

    expect(queryClient.getQueryData(queryKeys.user.me())).toEqual(testUser);
  });
});
