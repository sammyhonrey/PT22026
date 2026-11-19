import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Navigation } from "@/components/layout/Navigation";
import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { testUser } from "@/test/fixtures";
import { renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/auth/AuthProvider", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);

function authValue(isAuthenticated: boolean): AuthContextValue {
  return {
    isAuthenticated,
    initializationError: null,
    isLoading: false,
    retryInitialization: vi.fn(),
    signIn: vi.fn(async () => testUser),
    signOut: vi.fn(async () => undefined),
    signUp: vi.fn(async () => testUser),
    user: isAuthenticated ? testUser : null,
  };
}

beforeEach(() => mockedUseAuth.mockReset());

describe("Navigation", () => {
  it("shows only public destinations to guests", () => {
    mockedUseAuth.mockReturnValue(authValue(false));
    renderWithProviders(<Navigation />);

    expect(screen.getByRole("img", { name: "MovieMind" })).toHaveAttribute(
      "src",
      "/images/Glossy%20MovieMind%20Cinematic%20Logo.png",
    );
    expect(screen.getByRole("link", { name: "MovieMind home" })).toHaveAttribute("href", "/");
    expect(screen.getAllByRole("link", { name: "Home" })).toHaveLength(2);
    expect(screen.getAllByRole("link", { name: "Discover" })).toHaveLength(2);
    expect(screen.queryByRole("link", { name: "My List" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Profile" })).not.toBeInTheDocument();
  });

  it("reveals personal destinations after authentication", () => {
    mockedUseAuth.mockReturnValue(authValue(true));
    renderWithProviders(<Navigation />);

    for (const label of ["My Taste", "My List", "Ratings", "Profile", "Settings"]) {
      expect(screen.getAllByRole("link", { name: label }).length).toBeGreaterThan(0);
    }
  });

  it("keeps mobile navigation targets at least 44px tall", () => {
    mockedUseAuth.mockReturnValue(authValue(false));
    renderWithProviders(<Navigation />);

    const mobileNav = screen.getByRole("navigation", { name: "Mobile navigation" });
    for (const link of screen.getAllByRole("link", { name: /Home|Discover/ })) {
      if (mobileNav.contains(link)) expect(link.className).toContain("min-h-11");
    }
  });
});
