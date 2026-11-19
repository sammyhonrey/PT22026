import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { MovieCollection } from "@/features/discovery/components/MovieCollection";
import type { ApiClientError } from "@/lib/api/errors";
import { dune, interstellar } from "@/test/fixtures";
import { renderWithProviders } from "@/test/test-utils";

const emptyCopy = { emptyDescription: "Try another search.", emptyTitle: "No movies found" };

describe("MovieCollection", () => {
  it("renders the loading skeleton", () => {
    renderWithProviders(<MovieCollection {...emptyCopy} error={null} isLoading movies={undefined} onRetry={vi.fn()} />);

    expect(screen.getByRole("status", { name: "Loading movies" })).toBeInTheDocument();
  });

  it("renders the empty state when no movies are returned", () => {
    renderWithProviders(<MovieCollection {...emptyCopy} error={null} isLoading={false} movies={[]} onRetry={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "No movies found" })).toBeInTheDocument();
    expect(screen.getByText("Try another search.")).toBeInTheDocument();
  });

  it("renders a safe API error and retries on request", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const error: ApiClientError = {
      code: "internal_error",
      message: "Something went wrong on our side. Please try again shortly.",
      retryAfterSeconds: null,
      status: 500,
    };

    renderWithProviders(<MovieCollection {...emptyCopy} error={error} isLoading={false} movies={undefined} onRetry={onRetry} />);

    expect(screen.getByRole("alert")).toHaveTextContent(error.message);
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("does not expose an unexpected internal error message", () => {
    renderWithProviders(
      <MovieCollection {...emptyCopy} error={new Error("raw database stack trace")} isLoading={false} movies={undefined} onRetry={vi.fn()} />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("We could not load movies. Please try again.");
    expect(screen.queryByText(/raw database stack trace/)).not.toBeInTheDocument();
  });

  it("renders a movie grid for returned data", () => {
    renderWithProviders(
      <MovieCollection {...emptyCopy} error={null} isLoading={false} movies={[dune, interstellar]} onRetry={vi.fn()} />,
    );

    expect(screen.getByRole("heading", { name: dune.title })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: interstellar.title })).toBeInTheDocument();
  });
});
