import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MoviePoster } from "@/components/movies/MoviePoster";
import { getMoviePosterUrl } from "@/lib/movie-posters";

vi.mock("@/lib/movie-posters", () => ({ getMoviePosterUrl: vi.fn() }));

const mockedGetMoviePosterUrl = vi.mocked(getMoviePosterUrl);

describe("MoviePoster", () => {
  beforeEach(() => mockedGetMoviePosterUrl.mockReset());

  it("renders a lazy-loaded portrait image with meaningful alt text", () => {
    mockedGetMoviePosterUrl.mockReturnValue("https://image.tmdb.org/t/p/w342/poster.jpg");

    render(
      <MoviePoster
        className="aspect-[2/3]"
        movie={{ movielens_id: 1, title: "Dune" }}
      />,
    );

    const image = screen.getByRole("img", { name: "Poster for Dune" });
    expect(image).toHaveAttribute("loading", "lazy");
    expect(image).toHaveClass("object-cover");
    expect(image).toHaveAttribute("src", "https://image.tmdb.org/t/p/w342/poster.jpg");
    expect(image).toHaveClass("opacity-0");
    fireEvent.load(image);
    expect(image).toHaveClass("opacity-100");
  });

  it("shows the placeholder for missing and broken posters", () => {
    mockedGetMoviePosterUrl.mockReturnValue(null);
    const { rerender } = render(<MoviePoster movie={{ movielens_id: 1, title: "Dune" }} />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("Dune")).toBeInTheDocument();

    mockedGetMoviePosterUrl.mockReturnValue("https://image.tmdb.org/t/p/w342/broken.jpg");
    rerender(<MoviePoster movie={{ movielens_id: 1, title: "Dune" }} />);
    fireEvent.error(screen.getByRole("img", { name: "Poster for Dune" }));

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("Dune")).toBeInTheDocument();
  });
});
