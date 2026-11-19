import { apiClient } from "@/lib/api/client";
import { authSession } from "@/lib/auth/session";
import type { CsrfTokenResponse, LoginResponse, RegisterResponse, User } from "@/types/api";

export type LoginPayload = { email: string; password: string };
export type RegisterPayload = { display_name?: string; email: string; password: string; username?: string };

export async function login(payload: LoginPayload) {
  const { data } = await apiClient.post<LoginResponse>("/auth/login", payload);
  return data;
}

export async function register(payload: RegisterPayload) {
  const { data } = await apiClient.post<RegisterResponse>("/auth/register", payload);
  return data;
}

export async function fetchCurrentUser(signal?: AbortSignal) {
  const { data } = await apiClient.get<User>("/users/me", { signal });
  return data;
}

export async function fetchCsrfToken() {
  const { data } = await apiClient.get<CsrfTokenResponse>("/auth/csrf");
  authSession.setCsrfToken(data.csrf_token);
  return data.csrf_token;
}

export async function logout() {
  await apiClient.post("/auth/logout");
}

export async function updateCurrentUser(payload: { display_name?: string; username?: string }) {
  const { data } = await apiClient.patch<User>("/users/me", payload);
  return data;
}
