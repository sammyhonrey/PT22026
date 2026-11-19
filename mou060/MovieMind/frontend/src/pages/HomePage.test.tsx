import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Route, Routes } from "react-router-dom";

import { HomePage } from "@/pages/HomePage";
import { apiBaseUrl } from "@/lib/api/client";
import { server } from "@/test/msw/server";
import { batman, dune, interstellar, pageOf } from "@/test/fixtures";
import { LocationProbe, renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/recommendations/components/RecommendationPreview", () => ({
  RecommendationPreview: () => <aside>Your Recommendations</aside>,
}));
vi.mock("@/features/watchlist/components/WatchlistButton", () => ({
  WatchlistButton: ({ movieTitle }: { movieTitle: string }) => (
    <button aria-label={`Add ${movieTitle} to My List`} type="button" />
  ),
}));

beforeEach(() => {
  server.use(
    http.get(`${apiBaseUrl}/movies/popular`, () => HttpResponse.json([dune, batman])),
    http.get(`${apiBaseUrl}/movies`, () => HttpResponse.json(pageOf([interstellar], 8))),
  );
});

describe("HomePage", () => {
  it("renders the cinematic hero, real movie rails, recommendation panel, and working routes", async () => {
    renderWithProviders(
      <Routes>
        <Route element={<><HomePage /><LocationProbe /></>} path="/" />
        <Route element={<p>Discover destination</p>} path="/discover" />
      </Routes>,
    );

    expect(screen.getByRole("heading", { name: "Your Personal Movie Companion" })).toBeInTheDocument();
    expect(screen.getByText("Get AI-powered recommendations and discover movies you'll love!")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Start Exploring/ })).toHaveAttribute("href", "/discover");
    expect(screen.getByRole("link", { name: /Learn More/ })).toHaveAttribute("href", "/preferences");
    expect(await screen.findByRole("region", { name: "Trending Now" })).toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Top Rated Movies" })).toBeInTheDocument();
    expect(screen.getByRole("complementary")).toHaveTextContent("Your Recommendations");
    expect(screen.getByRole("heading", { name: "Ready for your next movie adventure?" })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "See All" })).toHaveLength(2);
    expect(screen.getByRole("link", { name: "Discover Movies" })).toHaveAttribute("href", "/discover");
    expect(screen.getByRole("button", { name: `Add ${dune.title} to My List` })).toBeInTheDocument();

    await userEvent.setup().click(screen.getByRole("link", { name: /Start Exploring/ }));
    expect(screen.getByText("Discover destination")).toBeInTheDocument();
  });
});
