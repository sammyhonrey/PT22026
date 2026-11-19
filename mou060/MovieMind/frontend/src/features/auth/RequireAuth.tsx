import { Navigate, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { PageLoader } from "@/components/ui/PageLoader";
import { useAuth } from "@/features/auth/AuthProvider";

export function RequireAuth({ children }: { children: ReactNode }) {
  const { initializationError, isAuthenticated, isLoading, retryInitialization } = useAuth();
  const location = useLocation();
  if (isLoading) return <PageLoader label="Restoring your MovieMind session…" />;
  if (initializationError) {
    return (
      <section className="mx-auto max-w-xl rounded-3xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6 text-center">
        <p className="text-sm text-[var(--mm-text-muted)]" role="alert">{initializationError}</p>
        <Button className="mt-4" onClick={retryInitialization} variant="secondary">Try again</Button>
      </section>
    );
  }
  if (!isAuthenticated) return <Navigate replace state={{ from: location }} to="/login" />;
  return <>{children}</>;
}

export function GuestOnly({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();
  if (isLoading) return <PageLoader label="Restoring your MovieMind session…" />;
  if (!isAuthenticated) return <>{children}</>;

  const state = location.state;
  if (typeof state === "object" && state !== null && "from" in state) {
    const from = state.from;
    if (
      typeof from === "object" &&
      from !== null &&
      "pathname" in from &&
      typeof from.pathname === "string" &&
      from.pathname.startsWith("/") &&
      !from.pathname.startsWith("//")
    ) {
      return (
        <Navigate
          replace
          to={{
            pathname: from.pathname,
            search: "search" in from && typeof from.search === "string" ? from.search : undefined,
            hash: "hash" in from && typeof from.hash === "string" ? from.hash : undefined,
          }}
        />
      );
    }
  }
  return <Navigate replace to="/" />;
}
