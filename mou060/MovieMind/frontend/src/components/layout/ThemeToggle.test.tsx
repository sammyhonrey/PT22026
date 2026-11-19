import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { ThemeProvider } from "@/features/theme/ThemeProvider";

function mockColorScheme(prefersLight: boolean) {
  vi.stubGlobal("matchMedia", vi.fn(() => ({
    matches: prefersLight,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  })));
}

function renderThemeToggle() {
  return render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>,
  );
}

describe("ThemeToggle", () => {
  beforeEach(() => {
    localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  it("uses the system preference when there is no saved choice", () => {
    mockColorScheme(true);
    renderThemeToggle();

    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(screen.getByRole("switch", { name: "Light mode" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByTitle("Switch to dark mode")).toBeInTheDocument();
    expect(localStorage.getItem("moviemind-theme")).toBeNull();
  });

  it("restores a saved theme instead of the system preference", () => {
    localStorage.setItem("moviemind-theme", "dark");
    mockColorScheme(true);
    renderThemeToggle();

    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    expect(screen.getByTitle("Switch to light mode")).toBeInTheDocument();
  });

  it("switches themes and persists the selection", () => {
    mockColorScheme(false);
    renderThemeToggle();

    fireEvent.click(screen.getByRole("switch", { name: "Light mode" }));

    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(localStorage.getItem("moviemind-theme")).toBe("light");
    expect(screen.getByTitle("Switch to dark mode")).toBeInTheDocument();
  });
});
