import { ArrowLeft, Eye, EyeOff, Film } from "lucide-react";
import { type FormEvent, type ReactNode, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Input } from "@/components/ui/Input";
import { isApiClientError } from "@/lib/api/errors";
import { useAuth } from "@/features/auth/AuthProvider";

type AuthMode = "login" | "register";
type FieldErrors = Partial<Record<"displayName" | "email" | "password" | "username", string>>;

const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const usernamePattern = /^[a-zA-Z0-9_.-]+$/;

function destination(state: unknown) {
  if (typeof state !== "object" || state === null || !("from" in state)) return "/";
  const from = (state as { from?: { pathname?: unknown; search?: unknown } }).from;
  if (!from || typeof from.pathname !== "string" || !from.pathname.startsWith("/") || from.pathname.startsWith("//")) return "/";
  return `${from.pathname}${typeof from.search === "string" ? from.search : ""}`;
}

export function AuthPage({ mode }: { mode: AuthMode }) {
  const isLogin = mode === "login";
  const { signIn, signUp } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  function validate() {
    const next: FieldErrors = {};
    if (!email.trim()) next.email = "Email is required.";
    else if (!emailPattern.test(email.trim())) next.email = "Enter a valid email address.";
    if (!password) next.password = "Password is required.";
    else if (!isLogin && password.length < 8) next.password = "Use at least 8 characters.";
    else if (password.length > 128) next.password = "Password cannot exceed 128 characters.";
    if (!isLogin && username.trim() && (username.trim().length < 3 || username.trim().length > 50 || !usernamePattern.test(username.trim()))) next.username = "Use 3–50 letters, numbers, dots, underscores, or hyphens.";
    if (!isLogin && displayName.trim().length > 100) next.displayName = "Display name cannot exceed 100 characters.";
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError(null);
    if (!validate()) return;
    setIsSubmitting(true);
    try {
      if (isLogin) await signIn({ email: email.trim(), password });
      else await signUp({
        display_name: displayName.trim() || undefined,
        email: email.trim(),
        password,
        username: username.trim() || undefined,
      });
      navigate(destination(location.state), { replace: true });
    } catch (error) {
      setFormError(isApiClientError(error) ? error.message : "We could not complete that request. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  const heading = isLogin ? "Welcome back" : "Create your account";
  const help = isLogin ? "Sign in to rate films, build your list, and personalize recommendations." : "Start rating movies to make MovieMind recommendations feel more personal.";

  return (
    <main className="auth-stage relative grid min-h-screen place-items-center p-5 text-[var(--mm-text)]">
      <div className="absolute right-5 top-5 sm:right-8 sm:top-8"><ThemeToggle /></div>
      <section className="w-full max-w-md rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface-glass)] p-7 shadow-[var(--mm-card-shadow)] backdrop-blur-xl sm:p-9">
        <Link className="inline-flex items-center gap-2 text-sm font-semibold text-[var(--mm-text-muted)] hover:text-[var(--mm-text)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]" to="/"><ArrowLeft aria-hidden="true" className="size-4" /> Back to Home</Link>
        <div className="mt-8 grid size-12 place-items-center rounded-2xl bg-linear-to-br from-[var(--mm-primary)] to-[var(--mm-primary-deep)]"><Film aria-hidden="true" className="size-7" /></div>
        <h1 className="mt-6 text-3xl font-black tracking-tight">{heading}</h1>
        <p className="mt-3 leading-6 text-[var(--mm-text-muted)]">{help}</p>
        <form className="mt-8 space-y-5" noValidate onSubmit={submit}>
          {!isLogin && <FormField error={errors.displayName} id="display-name" label="Display name"><Input aria-describedby={errors.displayName ? "display-name-error" : undefined} aria-invalid={Boolean(errors.displayName)} autoComplete="name" id="display-name" maxLength={100} onChange={(event) => setDisplayName(event.target.value)} value={displayName} /></FormField>}
          {!isLogin && <FormField error={errors.username} id="username" label="Username (optional)"><Input aria-describedby={errors.username ? "username-error" : undefined} aria-invalid={Boolean(errors.username)} autoComplete="username" id="username" maxLength={50} onChange={(event) => setUsername(event.target.value)} value={username} /></FormField>}
          <FormField error={errors.email} id="email" label="Email address"><Input aria-describedby={errors.email ? "email-error" : undefined} aria-invalid={Boolean(errors.email)} autoComplete="email" id="email" inputMode="email" onChange={(event) => setEmail(event.target.value)} type="email" value={email} /></FormField>
          <FormField error={errors.password} id="password" label={isLogin ? "Password" : "Password (8–128 characters)"}>
            <div className="relative"><Input aria-describedby={errors.password ? "password-error" : undefined} aria-invalid={Boolean(errors.password)} autoComplete={isLogin ? "current-password" : "new-password"} id="password" maxLength={128} onChange={(event) => setPassword(event.target.value)} type={showPassword ? "text" : "password"} value={password} className="pr-12" /><button aria-label={showPassword ? "Hide password" : "Show password"} className="absolute right-2 top-1/2 grid size-11 -translate-y-1/2 place-items-center rounded-lg text-[var(--mm-text-muted)] hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)] focus-visible:outline-2 focus-visible:outline-[var(--mm-accent)]" onClick={() => setShowPassword((visible) => !visible)} type="button">{showPassword ? <EyeOff aria-hidden="true" className="size-4" /> : <Eye aria-hidden="true" className="size-4" />}</button></div>
          </FormField>
          {formError && <p className="rounded-xl border border-[var(--mm-danger-soft-border)] bg-[var(--mm-danger-soft-bg)] px-3 py-2 text-sm text-[var(--mm-danger)]" role="alert">{formError}</p>}
          <Button className="w-full" isLoading={isSubmitting} type="submit">{isLogin ? "Sign in" : "Create account"}</Button>
        </form>
        <p className="mt-6 text-center text-sm text-[var(--mm-text-muted)]">{isLogin ? "New to MovieMind?" : "Already have an account?"} <Link className="font-semibold text-[var(--mm-accent)] hover:text-[var(--mm-text)]" to={isLogin ? "/register" : "/login"}>{isLogin ? "Create an account" : "Log in"}</Link></p>
      </section>
    </main>
  );
}

function FormField({ children, error, id, label }: { children: ReactNode; error?: string; id: string; label: string }) {
  return <div><label className="mb-2 block text-sm font-semibold text-[var(--mm-text)]" htmlFor={id}>{label}</label>{children}{error && <p className="mt-1.5 text-sm text-[var(--mm-danger)]" id={`${id}-error`}>{error}</p>}</div>;
}
