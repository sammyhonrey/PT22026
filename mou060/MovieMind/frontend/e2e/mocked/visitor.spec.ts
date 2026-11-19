import { expect, test } from "@playwright/test";

import { duneId, installMockApi } from "../support/mock-api";

test.beforeEach(async ({ page }) => {
  await installMockApi(page);
});

test("visitor discovers a movie and opens its detail page", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1, name: /Your Personal Movie Companion/ })).toBeVisible();
  const card = page.getByRole("link", { name: "View details for Dune" });
  await expect(card).toBeVisible();
  await card.click();

  await expect(page).toHaveURL(new RegExp(`/movies/${duneId}$`));
  await expect(page.getByRole("heading", { level: 1, name: "Dune" })).toBeVisible();
  await expect(page.getByText(/Paul Atreides travels to the desert planet Arrakis/)).toBeVisible();
});

test("visitor searches the catalogue with a shareable URL", async ({ page }) => {
  await page.goto("/search");
  const main = page.locator("#main-content");

  await main.getByRole("textbox", { name: "Search movies" }).fill("batman");
  await main.getByRole("button", { name: "Search" }).click();

  await expect(page).toHaveURL(/\/search\?q=batman$/);
  await expect(page.getByRole("heading", { name: /Results for/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "View details for Batman Begins" })).toBeVisible();
});

test("visitor can navigate to the login page", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("link", { name: "Log in" }).click();

  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { level: 1, name: "Welcome back" })).toBeVisible();
});
