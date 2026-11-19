import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { MemoryRouter, useLocation, type MemoryRouterProps } from "react-router-dom";

import { AuthProvider } from "@/features/auth/AuthProvider";
import { ThemeProvider } from "@/features/theme/ThemeProvider";

export function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      mutations: { retry: false },
      queries: { gcTime: Infinity, retry: false, staleTime: 0 },
    },
  });
}

type RenderWithProvidersOptions = Omit<RenderOptions, "wrapper"> & {
  initialEntries?: MemoryRouterProps["initialEntries"];
  queryClient?: QueryClient;
  withAuth?: boolean;
};

export function renderWithProviders(
  ui: ReactElement,
  {
    initialEntries = ["/"],
    queryClient = createTestQueryClient(),
    withAuth = false,
    ...renderOptions
  }: RenderWithProvidersOptions = {},
) {
  function Wrapper({ children }: { children: ReactNode }) {
    const router = <MemoryRouter initialEntries={initialEntries}>{children}</MemoryRouter>;
    return (
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          {withAuth ? <AuthProvider>{router}</AuthProvider> : router}
        </QueryClientProvider>
      </ThemeProvider>
    );
  }

  return { queryClient, ...render(ui, { wrapper: Wrapper, ...renderOptions }) };
}

export function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{`${location.pathname}${location.search}`}</output>;
}
