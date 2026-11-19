import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, type PropsWithChildren, useContext, useEffect, useMemo, useState } from "react";

import {
  fetchCsrfToken,
  fetchCurrentUser,
  login,
  logout,
  register,
  type LoginPayload,
  type RegisterPayload,
} from "@/features/auth/api";
import { isApiClientError } from "@/lib/api/errors";
import { refreshAccessToken } from "@/lib/api/client";
import { authSession } from "@/lib/auth/session";
import { queryKeys } from "@/lib/query/keys";
import type { User } from "@/types/api";

export type AuthContextValue = {
  isAuthenticated: boolean;
  initializationError: string | null;
  isLoading: boolean;
  retryInitialization: () => void;
  signIn: (payload: LoginPayload) => Promise<User>;
  signOut: () => Promise<void>;
  signUp: (payload: RegisterPayload) => Promise<User>;
  user: User | null;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const queryClient = useQueryClient();
  const [token, setToken] = useState(() => authSession.getToken());
  const [isInitializing, setIsInitializing] = useState(true);
  const [initializationError, setInitializationError] = useState<string | null>(null);
  const [initializationAttempt, setInitializationAttempt] = useState(0);
  const me = useQuery({
    queryKey: queryKeys.user.me(),
    queryFn: ({ signal }) => fetchCurrentUser(signal),
    enabled: Boolean(token),
    staleTime: 60_000,
    retry: false,
  });

  useEffect(() => {
    let active = true;
    const unsubscribe = authSession.subscribe((nextToken) => {
      setToken(nextToken);
      if (nextToken === null) {
        setIsInitializing(false);
        setInitializationError(null);
      }
    });

    async function restoreSession() {
      setIsInitializing(true);
      setInitializationError(null);
      try {
        if (!authSession.getToken()) {
          const restoredToken = await refreshAccessToken();
          if (!restoredToken) {
            if (active) {
              authSession.clear();
              queryClient.removeQueries({ queryKey: queryKeys.user.me() });
            }
            return;
          }
        }
        const user = await queryClient.fetchQuery({
          queryKey: queryKeys.user.me(),
          queryFn: () => fetchCurrentUser(),
          staleTime: 60_000,
        });
        if (active) queryClient.setQueryData(queryKeys.user.me(), user);
      } catch (error) {
        if (!active) return;
        if (isApiClientError(error) && (error.status === 401 || error.status === 403)) {
          authSession.clear();
          queryClient.removeQueries({ queryKey: queryKeys.user.me() });
        } else {
          setInitializationError(
            isApiClientError(error)
              ? error.message
              : "We could not restore your session. Check your connection and try again.",
          );
        }
      } finally {
        if (active) setIsInitializing(false);
      }
    }

    void restoreSession();
    return () => {
      active = false;
      unsubscribe();
    };
  }, [initializationAttempt, queryClient]);

  async function signIn(payload: LoginPayload) {
    const response = await login(payload);
    authSession.setSession(response.access_token, response.csrf_token);
    queryClient.setQueryData(queryKeys.user.me(), response.user);
    setInitializationError(null);
    setIsInitializing(false);
    return response.user;
  }

  async function signUp(payload: RegisterPayload) {
    const response = await register(payload);
    authSession.setSession(response.access_token, response.csrf_token);
    queryClient.setQueryData(queryKeys.user.me(), response.user);
    setInitializationError(null);
    setIsInitializing(false);
    return response.user;
  }

  async function signOut() {
    try {
      await fetchCsrfToken();
      await logout();
    } finally {
      authSession.clear();
      queryClient.removeQueries({ queryKey: queryKeys.user.me() });
      setInitializationError(null);
      setIsInitializing(false);
    }
  }

  function retryInitialization() {
    setInitializationAttempt((attempt) => attempt + 1);
  }

  const value = useMemo<AuthContextValue>(() => ({
    isAuthenticated: Boolean(token) && Boolean(me.data) && initializationError === null,
    initializationError,
    isLoading: isInitializing || (Boolean(token) && me.isLoading),
    retryInitialization,
    signIn,
    signOut,
    signUp,
    user: me.data ?? null,
  }), [initializationError, isInitializing, me.data, me.isLoading, token]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used within AuthProvider");
  return value;
}
