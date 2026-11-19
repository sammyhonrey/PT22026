import { expect, test, type Page } from "@playwright/test";

/**
 * These tests never run against a default environment. Provide a safe,
 * non-production deployment and a dedicated test account:
 *
 *   E2E_BASE_URL=https://preview.example.com
 *   E2E_EMAIL=qa-user@example.com
 *   E2E_PASSWORD=...
 *   npm run test:e2e:live
 *
 * The account is mutated (watchlist and rating) by these tests; never use a
 * shared or production account.
 */
const email = process.env.E2E_EMAIL;
const password = process.env.E2E_PASSWORD;

test.describe("live backend account", () => {
  test.skip(!email || !password, "Set E2E_EMAIL and E2E_PASSWORD for a dedicated non-production test account.");

  test.use({ baseURL: process.env.E2E_BASE_URL });

  test("login, movie actions, recommendations, and logout", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Email address").fill(email!);
    await page.getByLabel("Password", { exact: true }).fill(password!);
    await page.getByRole("button", { name: "Sign in" }).click();

    await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();

    await page.goto("/discover");
    const firstMovie = page.getByRole("link", { name: /^View details for/ }).first();
    await expect(firstMovie).toBeVisible();
    await firstMovie.click();
    await expect(page).toHaveURL(/\/movies\/[0-9a-f-]{36}$/);

    await toggleSavedMovie(page);
    await expectRatingFeedback(page);

    await page.goto("/recommendations");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page).toHaveURL(/\/login$/);
  });
});

async function toggleSavedMovie(page: Page) {
  const add = page.getByRole("button", { name: "Add to My List" });
  if (await add.isVisible()) {
    await add.click();
    await expect(page.getByRole("button", { name: "Saved to My List" })).toBeVisible();
    await page.getByRole("button", { name: "Saved to My List" }).click();
    await expect(add).toBeVisible();
    return;
  }

  const saved = page.getByRole("button", { name: "Saved to My List" });
  await saved.click();
  await expect(add).toBeVisible();
  await add.click();
  await expect(saved).toBeVisible();
  await saved.click();
  await expect(add).toBeVisible();
}

async function expectRatingFeedback(page: Page) {
  const submit = page.getByRole("radio", { name: "Rate 4.5 out of 5" });
  await submit.click();
  await expect(page.getByText(/Your rating of 4\.5 was saved\.|could not|Too many/)).toBeVisible();
}
