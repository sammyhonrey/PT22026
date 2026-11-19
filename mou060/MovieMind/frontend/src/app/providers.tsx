import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState, type PropsWithChildren } from "react";

import { App } from "@/app/App";
import { AuthProvider } from "@/features/auth/AuthProvider";
import { ThemeProvider } from "@/features/theme/ThemeProvider";
import { authSession } from "@/lib/auth/session";
import { isUserSpecificQuery } from "@/lib/query/keys";

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  });
}

type QueryProviderProps = PropsWithChildren<{ client?: QueryClient }>;

export function QueryProvider({ children, client }: QueryProviderProps) {
  const [ownedClient] = useState(createQueryClient);
  const queryClient = client ?? ownedClient;

  useEffect(
    () =>
      authSession.subscribe((token) => {
        if (token === null) {
          queryClient.removeQueries({
            predicate: (query) => isUserSpecificQuery(query.queryKey),
          });
        }
      }),
    [queryClient],
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

export function AppProviders() {
  return (
    <ThemeProvider>
      <QueryProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </QueryProvider>
    </ThemeProvider>
  );
}
