import { http, HttpResponse } from "msw";

import { apiBaseUrl } from "@/lib/api/client";
import {
  batman,
  dune,
  genres,
  makeRating,
  makeWatchlistItem,
  pageOf,
  testMovies,
  testUser,
} from "@/test/fixtures";

const notFound = { code: "not_found", detail: "Resource not found" };

/**
 * Happy-path handlers that mirror the verified FastAPI contract. Individual
 * tests override these with server.use() to exercise failures and edge cases.
 */
export const defaultHandlers = [
  http.post(`${apiBaseUrl}/auth/login`, async ({ request }) => {
    const body = (await request.json()) as { email: string; password: string };
    if (body.password === "wrong-password") {
      return HttpResponse.json({ code: "unauthorized", detail: "Invalid credentials" }, { status: 401 });
    }
    return HttpResponse.json({
      access_token: "test-token",
      csrf_token: "test-csrf-token",
      expires_in: 1800,
      token_type: "bearer",
      user: testUser,
    });
  }),
  http.post(`${apiBaseUrl}/auth/register`, () =>
    HttpResponse.json(
      {
        access_token: "test-token",
        csrf_token: "test-csrf-token",
        expires_in: 1800,
        token_type: "bearer",
        user: testUser,
      },
      { status: 201 },
    ),
  ),
  http.get(`${apiBaseUrl}/auth/csrf`, () => HttpResponse.json({ csrf_token: null })),
  http.post(`${apiBaseUrl}/auth/refresh`, () =>
    HttpResponse.json({ code: "unauthorized", detail: "No refresh session" }, { status: 401 }),
  ),
  http.post(`${apiBaseUrl}/auth/logout`, () => new HttpResponse(null, { status: 204 })),
  http.get(`${apiBaseUrl}/users/me`, () => HttpResponse.json(testUser)),
  http.patch(`${apiBaseUrl}/users/me`, async ({ request }) => {
    const body = (await request.json()) as { display_name?: string; username?: string };
    return HttpResponse.json({ ...testUser, ...body });
  }),

  // Static movie routes must precede /movies/:movieId.
  http.get(`${apiBaseUrl}/movies/genres`, () => HttpResponse.json(genres)),
  http.get(`${apiBaseUrl}/movies/popular`, ({ request }) => {
    const limit = Number(new URL(request.url).searchParams.get("limit") ?? 20);
    return HttpResponse.json(testMovies.slice(0, limit));
  }),
  http.get(`${apiBaseUrl}/movies`, ({ request }) => {
    const url = new URL(request.url);
    const offset = Number(url.searchParams.get("offset") ?? 0);
    const limit = Number(url.searchParams.get("limit") ?? 20);
    const query = url.searchParams.get("q")?.toLowerCase();
    const genre = url.searchParams.get("genre")?.toLowerCase();
    const yearParam = url.searchParams.get("year");
    const year = yearParam === null ? null : Number(yearParam);
    const items = testMovies.filter((movie) => {
      if (query && !movie.title.toLowerCase().includes(query)) return false;
      if (genre && !movie.genres.some((item) => item.toLowerCase() === genre)) return false;
      if (year !== null && movie.release_year !== year) return false;
      return true;
    });
    return HttpResponse.json(pageOf(items, limit, offset));
  }),
  http.get(`${apiBaseUrl}/movies/:movieId`, ({ params }) => {
    const movie = testMovies.find((item) => item.id === params.movieId);
    return movie ? HttpResponse.json(movie) : HttpResponse.json(notFound, { status: 404 });
  }),

  http.get(`${apiBaseUrl}/ratings/me/:movieId`, () =>
    HttpResponse.json({ code: "not_found", detail: "Rating not found" }, { status: 404 }),
  ),
  http.get(`${apiBaseUrl}/ratings/me`, ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(pageOf([], Number(url.searchParams.get("limit") ?? 20), Number(url.searchParams.get("offset") ?? 0)));
  }),
  http.post(`${apiBaseUrl}/ratings`, async ({ request }) => {
    const body = (await request.json()) as { movie_id: string; score: number };
    const movie = testMovies.find((item) => item.id === body.movie_id) ?? dune;
    return HttpResponse.json(makeRating(movie, body.score), { status: 201 });
  }),
  http.delete(`${apiBaseUrl}/ratings/me/:movieId`, () => new HttpResponse(null, { status: 204 })),

  http.get(`${apiBaseUrl}/watchlist`, () => HttpResponse.json(pageOf([]))),
  http.post(`${apiBaseUrl}/watchlist/:movieId`, ({ params }) => {
    const movie = testMovies.find((item) => item.id === params.movieId) ?? dune;
    return HttpResponse.json(makeWatchlistItem(movie), { status: 201 });
  }),
  http.delete(`${apiBaseUrl}/watchlist/:movieId`, () => new HttpResponse(null, { status: 204 })),

  http.get(`${apiBaseUrl}/recommendations/me`, () =>
    HttpResponse.json({
      is_fallback: true,
      items: [
        { movie: dune, reason: "Popular with MovieMind viewers", score: 0.9, source: "popularity" },
        { movie: batman, reason: "Frequently saved by viewers", score: 0.8, source: "popularity" },
      ],
      limit: 20,
      source: "popularity",
    }),
  ),
];

