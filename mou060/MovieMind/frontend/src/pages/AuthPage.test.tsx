import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MemoryRouterProps } from "react-router-dom";

import { useAuth } from "@/features/auth/AuthProvider";
import type { AuthContextValue } from "@/features/auth/AuthProvider";
import { AuthPage } from "@/pages/AuthPage";
import { testUser } from "@/test/fixtures";
import { LocationProbe, renderWithProviders } from "@/test/test-utils";

vi.mock("@/features/auth/AuthProvider", () => ({ useAuth: vi.fn() }));

const mockedUseAuth = vi.mocked(useAuth);
const signIn = vi.fn();
const signUp = vi.fn();

beforeEach(() => {
  signIn.mockReset();
  signUp.mockReset();
  mockedUseAuth.mockReturnValue({
    isAuthenticated: false,
    initializationError: null,
    isLoading: false,
    retryInitialization: vi.fn(),
    signIn,
    signOut: vi.fn(async () => undefined),
    signUp,
    user: null,
  } satisfies AuthContextValue);
});

function renderAuth(mode: "login" | "register", initialEntries: MemoryRouterProps["initialEntries"] = ["/login"]) {
  return renderWithProviders(
    <>
      <AuthPage mode={mode} />
      <LocationProbe />
    </>,
    { initialEntries },
  );
}

describe("AuthPage", () => {
  it("blocks empty login submission with inline field errors", async () => {
    const user = userEvent.setup();
    renderAuth("login");

    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Email is required.")).toBeInTheDocument();
    expect(screen.getByText("Password is required.")).toBeInTheDocument();
    expect(signIn).not.toHaveBeenCalled();
  });

  it("rejects an invalid email before calling the API", async () => {
    const user = userEvent.setup();
    renderAuth("login");

    await user.type(screen.getByLabelText("Email address"), "not-an-email");
    await user.type(screen.getByLabelText("Password"), "secret");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Enter a valid email address.")).toBeInTheDocument();
    expect(signIn).not.toHaveBeenCalled();
  });

  it("submits trimmed credentials and returns to the requested route", async () => {
    const user = userEvent.setup();
    signIn.mockResolvedValue(testUser);
    renderAuth("login", [{ pathname: "/login", state: { from: { pathname: "/watchlist", search: "" } } }]);

    await user.type(screen.getByLabelText("Email address"), "  viewer@example.com  ");
    await user.type(screen.getByLabelText("Password"), "correct-horse");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() =>
      expect(signIn).toHaveBeenCalledWith({ email: "viewer@example.com", password: "correct-horse" }),
    );
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/watchlist"));
  });

  it("prevents duplicate submissions while a login request is pending", async () => {
    const user = userEvent.setup();
    let resolveSignIn!: (value: typeof testUser) => void;
    signIn.mockReturnValue(new Promise((resolve) => { resolveSignIn = resolve; }));
    renderAuth("login");

    await user.type(screen.getByLabelText("Email address"), "viewer@example.com");
    await user.type(screen.getByLabelText("Password"), "correct-horse");
    const submit = screen.getByRole("button", { name: "Sign in" });
    await user.click(submit);

    await waitFor(() => expect(submit).toBeDisabled());
    await user.click(submit);
    expect(signIn).toHaveBeenCalledTimes(1);

    resolveSignIn(testUser);
    await waitFor(() => expect(screen.getByTestId("location")).toHaveTextContent("/"));
  });

  it("shows a safe request error and preserves the entered email", async () => {
    const user = userEvent.setup();
    signIn.mockRejectedValue({
      code: "internal_error",
      message: "Something went wrong on our side. Please try again shortly.",
      retryAfterSeconds: null,
      status: 500,
    });
    renderAuth("login");

    await user.type(screen.getByLabelText("Email address"), "viewer@example.com");
    await user.type(screen.getByLabelText("Password"), "correct-horse");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong on our side.");
    expect(screen.getByLabelText("Email address")).toHaveValue("viewer@example.com");
  });

  it("validates registration-only fields and password length", async () => {
    const user = userEvent.setup();
    renderAuth("register", ["/register"]);

    expect(screen.getByLabelText("Display name")).toBeInTheDocument();
    expect(screen.getByLabelText("Username (optional)")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Email address"), "new@example.com");
    await user.type(screen.getByLabelText("Password (8–128 characters)"), "short");
    await user.click(screen.getByRole("button", { name: "Create account" }));

    expect(await screen.findByText("Use at least 8 characters.")).toBeInTheDocument();
    expect(signUp).not.toHaveBeenCalled();
  });
});
