import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";

import { fetchCurrentUser, login, register, updateCurrentUser } from "@/features/auth/api";
import { fetchGenres, fetchMovie, fetchMovies, fetchPopularMovies } from "@/features/discovery/api";
import { fetchMyRating, fetchMyRatings, upsertRating, deleteRating } from "@/features/ratings/api";
import { fetchMyRecommendations } from "@/features/recommendations/api";
import { addToWatchlist, fetchWatchlist, removeFromWatchlist } from "@/features/watchlist/api";
import { apiBaseUrl } from "@/lib/api/client";
import { authSession } from "@/lib/auth/session";
import { dune, makeRating, pageOf, testMovies, testUser } from "@/test/fixtures";
import { server } from "@/test/msw/server";

beforeEach(() => authSession.setToken("test-token"));

describe("auth endpoints", () => {
  it("posts the exact login schema", async () => {
    let body: unknown;
    server.use(
      http.post(`${apiBaseUrl}/auth/login`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({
          access_token: "token",
          csrf_token: "csrf",
          expires_in: 1800,
          token_type: "bearer",
          user: testUser,
        });
      }),
    );

    await login({ email: "viewer@example.com", password: "secret" });

    expect(body).toEqual({ email: "viewer@example.com", password: "secret" });
  });

  it("posts the accepted registration fields and patch schema", async () => {
    const registrations: unknown[] = [];
    const updates: unknown[] = [];
    server.use(
      http.post(`${apiBaseUrl}/auth/register`, async ({ request }) => {
        registrations.push(await request.json());
        return HttpResponse.json({
          access_token: "token",
          csrf_token: "csrf",
          expires_in: 1800,
          token_type: "bearer",
          user: testUser,
        }, { status: 201 });
      }),
      http.patch(`${apiBaseUrl}/users/me`, async ({ request }) => {
        updates.push(await request.json());
        return HttpResponse.json(testUser);
      }),
    );

    await register({ display_name: "Test Viewer", email: "new@example.com", password: "password123", username: "viewer" });
    await updateCurrentUser({ display_name: "Updated", username: "updated" });

    expect(registrations[0]).toEqual({
      display_name: "Test Viewer",
      email: "new@example.com",
      password: "password123",
      username: "viewer",
    });
    expect(updates[0]).toEqual({ display_name: "Updated", username: "updated" });
  });

  it("loads the current user from /users/me", async () => {
    let authorization: string | null = null;
    server.use(
      http.get(`${apiBaseUrl}/users/me`, ({ request }) => {
        authorization = request.headers.get("authorization");
        return HttpResponse.json(testUser);
      }),
    );

    await expect(fetchCurrentUser()).resolves.toEqual(testUser);
    expect(authorization).toBe("Bearer test-token");
  });
});

describe("discovery endpoints", () => {
  it("sends the discovered movie-list query parameters", async () => {
    let url: URL | undefined;
    server.use(
      http.get(`${apiBaseUrl}/movies`, ({ request }) => {
        url = new URL(request.url);
        return HttpResponse.json(pageOf(testMovies));
      }),
    );

    await fetchMovies({
      direction: "desc",
      genre: "Action",
      limit: 20,
      offset: 40,
      q: "dune",
      sort: "average_rating",
      year: 2021,
    });

    expect(url?.pathname).toBe("/api/v1/movies");
    expect(Object.fromEntries(url?.searchParams ?? [])).toEqual({
      direction: "desc",
      genre: "Action",
      limit: "20",
      offset: "40",
      q: "dune",
      sort: "average_rating",
      year: "2021",
    });
  });

  it("uses the static genre, popular, and detail paths", async () => {
    const paths: string[] = [];
    server.use(
      http.get(`${apiBaseUrl}/movies/genres`, ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json([]);
      }),
      http.get(`${apiBaseUrl}/movies/popular`, ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json(testMovies.slice(0, 1));
      }),
      http.get(`${apiBaseUrl}/movies/:movieId`, ({ request }) => {
        paths.push(new URL(request.url).pathname);
        return HttpResponse.json(dune);
      }),
    );

    await fetchGenres();
    await fetchPopularMovies(5);
    await fetchMovie(dune.id);

    expect(paths).toEqual(["/api/v1/movies/genres", "/api/v1/movies/popular", `/api/v1/movies/${dune.id}`]);
  });
});

describe("ratings endpoints", () => {
  it("reads the current user's rating and paginated list", async () => {
    const urls: string[] = [];
    server.use(
      http.get(`${apiBaseUrl}/ratings/me/:movieId`, ({ request }) => {
        urls.push(new URL(request.url).pathname);
        return HttpResponse.json(makeRating(dune, 4));
      }),
      http.get(`${apiBaseUrl}/ratings/me`, ({ request }) => {
        urls.push(new URL(request.url).pathname + new URL(request.url).search);
        return HttpResponse.json(pageOf([makeRating(dune, 4)]));
      }),
    );

    await fetchMyRating(dune.id);
    await fetchMyRatings(20, 10);

    expect(urls).toEqual([`/api/v1/ratings/me/${dune.id}`, "/api/v1/ratings/me?limit=10&offset=20"]);
  });

  it("upserts a rating and deletes with the documented endpoints", async () => {
    let body: unknown;
    let authorization: string | null = null;
    let deletedPath: string | undefined;
    server.use(
      http.post(`${apiBaseUrl}/ratings`, async ({ request }) => {
        body = await request.json();
        authorization = request.headers.get("authorization");
        return HttpResponse.json(makeRating(dune, 4.5), { status: 201 });
      }),
      http.delete(`${apiBaseUrl}/ratings/me/:movieId`, ({ request }) => {
        deletedPath = new URL(request.url).pathname;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    await upsertRating({ movie_id: dune.id, score: 4.5 });
    await deleteRating(dune.id);

    expect(body).toEqual({ movie_id: dune.id, score: 4.5 });
    expect(authorization).toBe("Bearer test-token");
    expect(deletedPath).toBe(`/api/v1/ratings/me/${dune.id}`);
  });
});

describe("watchlist endpoints", () => {
  it("lists, adds, and removes saved movies", async () => {
    const requests: Array<{ method: string; path: string; authorization: string | null }> = [];
    server.use(
      http.get(`${apiBaseUrl}/watchlist`, ({ request }) => {
        requests.push({ method: request.method, path: new URL(request.url).pathname + new URL(request.url).search, authorization: request.headers.get("authorization") });
        return HttpResponse.json(pageOf([]));
      }),
      http.post(`${apiBaseUrl}/watchlist/:movieId`, ({ request }) => {
        requests.push({ method: request.method, path: new URL(request.url).pathname, authorization: request.headers.get("authorization") });
        return HttpResponse.json({ created_at: "2025-02-01T12:00:00Z", id: "item", movie: dune }, { status: 201 });
      }),
      http.delete(`${apiBaseUrl}/watchlist/:movieId`, ({ request }) => {
        requests.push({ method: request.method, path: new URL(request.url).pathname, authorization: request.headers.get("authorization") });
        return new HttpResponse(null, { status: 204 });
      }),
    );

    await fetchWatchlist(20, 10);
    await addToWatchlist(dune.id);
    await removeFromWatchlist(dune.id);

    expect(requests).toEqual([
      { method: "GET", path: "/api/v1/watchlist?limit=10&offset=20", authorization: "Bearer test-token" },
      { method: "POST", path: `/api/v1/watchlist/${dune.id}`, authorization: "Bearer test-token" },
      { method: "DELETE", path: `/api/v1/watchlist/${dune.id}`, authorization: "Bearer test-token" },
    ]);
  });
});

describe("recommendations endpoints", () => {
  it("requests the authenticated recommendation limit", async () => {
    let url: URL | undefined;
    let authorization: string | null = null;
    server.use(
      http.get(`${apiBaseUrl}/recommendations/me`, ({ request }) => {
        url = new URL(request.url);
        authorization = request.headers.get("authorization");
        return HttpResponse.json({ is_fallback: true, items: [], limit: 10, source: "popularity" });
      }),
    );

    await fetchMyRecommendations(10);

    expect(url?.pathname).toBe("/api/v1/recommendations/me");
    expect(url?.searchParams.get("limit")).toBe("10");
    expect(authorization).toBe("Bearer test-token");
  });
});
