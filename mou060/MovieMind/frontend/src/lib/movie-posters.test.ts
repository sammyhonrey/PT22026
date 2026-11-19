import { describe, expect, it, vi } from "vitest";

vi.mock("@/data/movie-posters.json", () => ({
  default: {
    moviePosters: [
      { movieId: 1, tmdbId: 101, posterUrl: "https://image.tmdb.org/t/p/w342/poster.jpg" },
      { movieId: 2, tmdbId: null, posterUrl: null },
      { movieId: 3, tmdbId: 103, posterUrl: "javascript:alert(1)" },
      { movieId: 4, tmdbId: 104, posterUrl: "http://image.tmdb.org/poster.jpg" },
      { movieId: 5, tmdbId: 105, posterUrl: "https://example.com/poster.jpg" },
    ],
  },
}));

import { getMoviePosterUrl } from "@/lib/movie-posters";

describe("getMoviePosterUrl", () => {
  it("looks up posters by MovieLens movie ID and rejects missing or unsafe URLs", () => {
    expect(getMoviePosterUrl(1)).toBe("https://image.tmdb.org/t/p/w342/poster.jpg");
    expect(getMoviePosterUrl(2)).toBeNull();
    expect(getMoviePosterUrl(3)).toBeNull();
    expect(getMoviePosterUrl(4)).toBeNull();
    expect(getMoviePosterUrl(5)).toBeNull();
    expect(getMoviePosterUrl(null)).toBeNull();
    expect(getMoviePosterUrl(Number.NaN)).toBeNull();
  });
});
