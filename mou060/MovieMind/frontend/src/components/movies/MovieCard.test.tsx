import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MovieCard } from "@/components/movies/MovieCard";
import { getMoviePosterUrl } from "@/lib/movie-posters";
import { batman, dune } from "@/test/fixtures";
import { renderWithProviders } from "@/test/test-utils";

vi.mock("@/lib/movie-posters", () => ({ getMoviePosterUrl: vi.fn() }));
vi.mock("@/features/watchlist/components/WatchlistButton", () => ({
  WatchlistButton: ({ movieTitle }: { movieTitle: string }) => (
    <button aria-label={`Add ${movieTitle} to My List`} type="button" />
  ),
}));

const mockedGetMoviePosterUrl = vi.mocked(getMoviePosterUrl);

beforeEach(() => mockedGetMoviePosterUrl.mockReset());

describe("MovieCard", () => {
  it("renders confirmed API fields, the poster, and a details link", () => {
    mockedGetMoviePosterUrl.mockReturnValue("https://image.tmdb.org/t/p/w342/dune.jpg");
    renderWithProviders(<MovieCard movie={dune} />);

    expect(screen.getByRole("heading", { name: dune.title })).toBeInTheDocument();
    expect(screen.getByText(String(dune.release_year))).toBeInTheDocument();
    expect(screen.getByText(dune.average_rating!.toFixed(1))).toBeInTheDocument();
    expect(screen.getByText("Science Fiction · Adventure")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: `Poster for ${dune.title}` })).toHaveAttribute("src", "https://image.tmdb.org/t/p/w342/dune.jpg");
    expect(screen.getByRole("link", { name: `View details for ${dune.title}` })).toHaveAttribute("href", `/movies/${dune.id}`);
  });

  it("renders a poster fallback while keeping the available metadata", () => {
    mockedGetMoviePosterUrl.mockReturnValue(null);
    renderWithProviders(<MovieCard movie={batman} />);

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getAllByText(batman.title).length).toBeGreaterThan(0);
    expect(screen.getByText("2005")).toBeInTheDocument();
    expect(screen.getByText("8.2")).toBeInTheDocument();
  });

  it("can show a separate watchlist action without nesting it in the movie link", () => {
    mockedGetMoviePosterUrl.mockReturnValue(null);
    renderWithProviders(<MovieCard movie={dune} showWatchlistButton />);

    expect(screen.getByRole("button", { name: `Add ${dune.title} to My List` })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: `View details for ${dune.title}` })).toHaveAttribute(
      "href",
      `/movies/${dune.id}`,
    );
  });

  it("handles a movie with no year, rating, genres, or poster", () => {
    const unavailable = {
      ...dune,
      average_rating: null,
      genres: [],
      poster_url: null,
      ratings_count: null,
      release_year: null,
    };

    mockedGetMoviePosterUrl.mockReturnValue(null);
    renderWithProviders(<MovieCard movie={unavailable} />);

    expect(screen.getByText("Year unavailable")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByText("·")).not.toBeInTheDocument();
  });
});
