import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll } from "vitest";

import { authSession } from "@/lib/auth/session";
import { server } from "@/test/msw/server";

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));

afterEach(() => {
  cleanup();
  document.documentElement.removeAttribute("data-theme");
  server.resetHandlers();
  authSession.clear();
  localStorage.clear();
  sessionStorage.clear();
});

afterAll(() => server.close());
