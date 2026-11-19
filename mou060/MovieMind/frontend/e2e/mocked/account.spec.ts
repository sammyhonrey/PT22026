import { expect, test, type Page } from "@playwright/test";

import { duneId, installMockApi } from "../support/mock-api";

test.beforeEach(async ({ page }) => {
  await installMockApi(page);
});

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email address").fill("viewer@example.com");
  await page.getByLabel("Password", { exact: true }).fill("correct-horse");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
}

test("signed-in user saves, rates, views recommendations, and signs out", async ({ page }) => {
  await login(page);

  await page.goto("/watchlist");
  await page.reload();
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();

  await page.goto(`/movies/${duneId}`);
  const add = page.getByRole("button", { name: "Add to My List" });
  await expect(add).toBeVisible();
  await add.click();
  await expect(page.getByRole("button", { name: "Saved to My List" })).toBeVisible();
  await page.getByRole("button", { name: "Saved to My List" }).click();
  await expect(page.getByRole("button", { name: "Add to My List" })).toBeVisible();

  await page.getByRole("radio", { name: "Rate 4.5 out of 5" }).click();
  await expect(page.getByText("Your rating of 4.5 was saved.")).toBeVisible();

  await page.goto("/recommendations");
  await expect(page.getByRole("heading", { level: 1, name: "Popular picks while you get started" })).toBeVisible();

  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { level: 1, name: "Welcome back" })).toBeVisible();
});
