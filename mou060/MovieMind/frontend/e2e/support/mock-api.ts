import type { Page, Route } from "@playwright/test";

type Movie = {
  average_rating: number | null;
  genres: string[];
  id: string;
  movielens_id: number | null;
  overview: string | null;
  poster_url: string | null;
  ratings_count: number | null;
  release_year: number | null;
  title: string;
};

const user = {
  created_at: "2025-01-15T10:00:00Z",
  display_name: "Test Viewer",
  email: "viewer@example.com",
  id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  is_active: true,
  role: "user",
  username: "viewer",
};

export const movies: Movie[] = [
  {
    average_rating: 8.4,
    genres: ["Science Fiction", "Adventure"],
    id: "11111111-1111-4111-8111-111111111111",
    movielens_id: 1,
    overview: "Paul Atreides travels to the desert planet Arrakis to protect his family and his people.",
    poster_url: null,
    ratings_count: 120,
    release_year: 2021,
    title: "Dune",
  },
  {
    average_rating: 8.2,
    genres: ["Action", "Crime"],
    id: "22222222-2222-4222-8222-222222222222",
    movielens_id: 2,
    overview: "Bruce Wayne becomes Batman and confronts the crime that shaped Gotham.",
    poster_url: null,
    ratings_count: 98,
    release_year: 2005,
    title: "Batman Begins",
  },
  {
    average_rating: 8.7,
    genres: ["Science Fiction", "Drama"],
    id: "33333333-3333-4333-8333-333333333333",
    movielens_id: 3,
    overview: "A team of explorers travels through a wormhole in an attempt to save humanity.",
    poster_url: null,
    ratings_count: 145,
    release_year: 2014,
    title: "Interstellar",
  },
];

export const duneId = movies[0].id;

type MockState = {
  ratings: Map<string, number>;
  signedIn: boolean;
  watchlist: Set<string>;
};

const responseHeaders = {
  "access-control-allow-headers": "authorization,content-type,x-csrf-token",
  "access-control-allow-methods": "GET,POST,PATCH,DELETE,OPTIONS",
  "access-control-allow-origin": "*",
};

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    body: JSON.stringify(body),
    contentType: "application/json",
    headers: responseHeaders,
    status,
  });
}

function pageOf<T>(items: T[], limit = 20, offset = 0) {
  return { items: items.slice(offset, offset + limit), limit, offset, total: items.length };
}

function ratingFor(movie: Movie, score: number) {
  return {
    created_at: "2025-02-01T12:00:00Z",
    id: `rating-${movie.id}`,
    movie,
    movie_id: movie.id,
    score,
    updated_at: "2025-02-01T12:00:00Z",
  };
}

function watchlistItem(movie: Movie) {
  return { created_at: "2025-02-01T12:00:00Z", id: `watchlist-${movie.id}`, movie };
}

/**
 * Install a stateful mock of the verified FastAPI contract for one page.
 * The app runs with VITE_API_BASE_URL pointing at the Vite server, so all
 * requests are same-origin and require no real backend or CORS configuration.
 */
export async function installMockApi(page: Page): Promise<MockState> {
  const state: MockState = { ratings: new Map(), signedIn: false, watchlist: new Set() };

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const method = request.method();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api\/v1/, "");

    if (method === "OPTIONS") {
      await route.fulfill({ headers: responseHeaders, status: 204 });
      return;
    }

    if (method === "POST" && path === "/auth/login") {
      state.signedIn = true;
      await json(route, {
        access_token: "e2e-token",
        csrf_token: "e2e-csrf-token",
        expires_in: 1800,
        token_type: "bearer",
        user,
      });
      return;
    }

    if (method === "POST" && path === "/auth/register") {
      state.signedIn = true;
      await json(route, {
        access_token: "e2e-token",
        csrf_token: "e2e-csrf-token",
        expires_in: 1800,
        token_type: "bearer",
        user,
      }, 201);
      return;
    }

    if (method === "GET" && path === "/auth/csrf") {
      return json(route, { csrf_token: state.signedIn ? "e2e-csrf-token" : null });
    }

    if (method === "POST" && path === "/auth/refresh") {
      if (!state.signedIn) return json(route, { code: "unauthorized", detail: "No session" }, 401);
      return json(route, {
        access_token: "e2e-token",
        csrf_token: "e2e-csrf-token",
        expires_in: 1800,
        token_type: "bearer",
      });
    }

    if (method === "POST" && path === "/auth/logout") {
      state.signedIn = false;
      return route.fulfill({ headers: responseHeaders, status: 204 });
      return;
    }

    if (method === "GET" && path === "/users/me") {
      if (!state.signedIn) return json(route, { code: "unauthorized", detail: "Sign in first" }, 401);
      return json(route, user);
    }

    if (method === "PATCH" && path === "/users/me") {
      await json(route, { ...user, ...(request.postDataJSON() as object) });
      return;
    }

    if (method === "GET" && path === "/movies/genres") {
      await json(route, [{ id: 1, name: "Action" }, { id: 2, name: "Drama" }]);
      return;
    }

    if (method === "GET" && path === "/movies/popular") {
      const limit = Number(url.searchParams.get("limit") ?? 20);
      await json(route, movies.slice(0, limit));
      return;
    }

    if (method === "GET" && path === "/movies") {
      const query = url.searchParams.get("q")?.toLowerCase();
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const limit = Number(url.searchParams.get("limit") ?? 20);
      const items = movies.filter((movie) => !query || movie.title.toLowerCase().includes(query));
      await json(route, pageOf(items, limit, offset));
      return;
    }

    const movieMatch = path.match(/^\/movies\/([^/]+)$/);
    if (method === "GET" && movieMatch) {
      const movie = movies.find((item) => item.id === movieMatch[1]);
      if (!movie) return json(route, { code: "not_found", detail: "Movie not found" }, 404);
      return json(route, movie);
    }

    const ratingMatch = path.match(/^\/ratings\/me\/([^/]+)$/);
    if (ratingMatch) {
      const score = state.ratings.get(ratingMatch[1]);
      if (method === "GET") {
        const movie = movies.find((item) => item.id === ratingMatch[1]);
        if (!movie || score === undefined) {
          return json(route, { code: "not_found", detail: "Rating not found" }, 404);
        }
        return json(route, ratingFor(movie, score));
      }
      if (method === "DELETE") {
        state.ratings.delete(ratingMatch[1]);
        return route.fulfill({ headers: responseHeaders, status: 204 });
      }
    }

    if (method === "GET" && path === "/ratings/me") {
      const items = [...state.ratings.entries()]
        .map(([movieId, score]) => {
          const movie = movies.find((item) => item.id === movieId);
          return movie ? ratingFor(movie, score) : null;
        })
        .filter((value) => value !== null);
      return json(route, pageOf(items));
    }

    if (method === "POST" && path === "/ratings") {
      const body = request.postDataJSON() as { movie_id: string; score: number };
      state.ratings.set(body.movie_id, body.score);
      const movie = movies.find((item) => item.id === body.movie_id) ?? movies[0];
      return json(route, ratingFor(movie, body.score), 201);
    }

    if (method === "GET" && path === "/watchlist") {
      const items = [...state.watchlist]
        .map((movieId) => movies.find((item) => item.id === movieId))
        .filter((movie): movie is Movie => Boolean(movie))
        .map(watchlistItem);
      return json(route, pageOf(items, 100));
    }

    const watchlistMatch = path.match(/^\/watchlist\/([^/]+)$/);
    if (watchlistMatch) {
      const movie = movies.find((item) => item.id === watchlistMatch[1]);
      if (method === "POST" && movie) {
        const existed = state.watchlist.has(movie.id);
        state.watchlist.add(movie.id);
        return json(route, watchlistItem(movie), existed ? 200 : 201);
      }
      if (method === "DELETE") {
        state.watchlist.delete(watchlistMatch[1]);
        return route.fulfill({ headers: responseHeaders, status: 204 });
      }
    }

    if (method === "GET" && path === "/recommendations/me") {
      return json(route, {
        is_fallback: state.ratings.size < 5,
        items: [
          { movie: movies[0], reason: "Popular with MovieMind viewers", score: 0.9, source: "popularity" },
          { movie: movies[1], reason: "Frequently saved by viewers", score: 0.8, source: "popularity" },
        ],
        limit: Number(url.searchParams.get("limit") ?? 20),
        source: "popularity",
      });
    }

    return json(route, { code: "not_found", detail: `No mock for ${method} ${path}` }, 404);
  });

  return state;
}
