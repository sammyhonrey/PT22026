import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";

import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { apiBaseUrl } from "@/lib/api/client";
import { RecommendationPreview } from "@/features/recommendations/components/RecommendationPreview";
import { dune, testUser } from "@/test/fixtures";
import { server } from "@/test/msw/server";
import { renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/auth/AuthProvider", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);

beforeEach(() => {
  mockedUseAuth.mockReset();
});

describe("RecommendationPreview", () => {
  it("renders API recommendation details, poster, rating, and movie navigation", async () => {
    const auth: AuthContextValue = {
      isAuthenticated: true,
      initializationError: null,
      isLoading: false,
      retryInitialization: vi.fn(),
      signIn: vi.fn(async () => testUser),
      signOut: vi.fn(async () => undefined),
      signUp: vi.fn(async () => testUser),
      user: testUser,
    };
    mockedUseAuth.mockReturnValue(auth);
    server.use(
      http.get(`${apiBaseUrl}/preferences/me`, () => HttpResponse.json({
        created_at: "2025-01-01T00:00:00Z",
        description: null,
        favorite_genres: ["Science Fiction"],
        favorite_movie_ids: [],
        updated_at: "2025-01-01T00:00:00Z",
        user_id: testUser.id,
      })),
      http.get(`${apiBaseUrl}/recommendations/me`, () => HttpResponse.json({
        is_fallback: false,
        items: [{ movie: dune, reason: "Matches your favorite genre.", score: 0.8, source: "Preference-based" }],
        limit: 3,
        source: "Preference-based",
      })),
    );

    renderWithProviders(<RecommendationPreview />);

    expect(screen.getByRole("status", { name: "Finding movies you'll love" })).toBeInTheDocument();
    expect(await screen.findByText("Personalized from your taste profile", {}, { timeout: 10_000 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Your Recommendations" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: `Poster for ${dune.title}` })).toBeInTheDocument();
    expect(screen.getByText(dune.average_rating!.toFixed(1))).toBeInTheDocument();
    expect(screen.getByRole("link", { name: `View details for ${dune.title}` })).toHaveAttribute(
      "href",
      `/movies/${dune.id}`,
    );
    expect(screen.getByRole("link", { name: "See all recommendations" })).toHaveAttribute(
      "href",
      "/recommendations",
    );
  }, 15_000);
});
