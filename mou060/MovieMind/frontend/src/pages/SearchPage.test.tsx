import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { SearchPage } from "@/pages/SearchPage";
import { apiBaseUrl } from "@/lib/api/client";
import { dune, pageOf } from "@/test/fixtures";
import { server } from "@/test/msw/server";
import { LocationProbe, renderWithProviders } from "@/test/test-utils";

describe("SearchPage", () => {
  it("shows a discovery prompt when no query is present", () => {
    renderWithProviders(<SearchPage />, { initialEntries: ["/search"] });

    expect(screen.getByRole("heading", { name: "Search Movies" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Search for a movie to get started" })).toBeInTheDocument();
  });

  it("restores a query from the URL and renders backend result counts", async () => {
    renderWithProviders(<SearchPage />, { initialEntries: ["/search?q=interstellar"] });

    expect(screen.getByLabelText("Search movies")).toHaveValue("interstellar");
    expect(await screen.findByRole("heading", { name: "Results for “interstellar”" })).toBeInTheDocument();
    expect(await screen.findByText("1 movie")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Interstellar" })).toBeInTheDocument();
  });

  it("submits an explicit search and syncs the query with the URL", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <>
        <SearchPage />
        <LocationProbe />
      </>,
      { initialEntries: ["/search"] },
    );

    await user.type(screen.getByLabelText("Search movies"), "batman");
    await user.click(screen.getByRole("button", { name: "Search" }));

    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/search?q=batman"));
    expect(await screen.findByRole("heading", { name: "Results for “batman”" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Batman Begins" })).toBeInTheDocument();
  });

  it("shows a no-results state for a query with no matches", async () => {
    renderWithProviders(<SearchPage />, { initialEntries: ["/search?q=zzzz-not-a-movie"] });

    expect(await screen.findByRole("heading", { name: "No movies found" })).toBeInTheDocument();
  });

  it("shows a loading button and movie skeletons while a search is pending", async () => {
    server.use(
      http.get(`${apiBaseUrl}/movies`, async () => {
        await delay(100);
        return HttpResponse.json(pageOf([dune]));
      }),
    );

    renderWithProviders(<SearchPage />, { initialEntries: ["/search?q=dune"] });

    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
    expect(screen.getByRole("status", { name: "Loading movies" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: dune.title })).toBeInTheDocument();
  });
});
