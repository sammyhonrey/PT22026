import { Search } from "lucide-react";
import { type FormEvent, useState } from "react";
import { Link, Outlet, useNavigate } from "react-router-dom";

import { Navigation } from "@/components/layout/Navigation";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { useAuth } from "@/features/auth/AuthProvider";

export function AppShell() {
  const [query, setQuery] = useState("");
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { isAuthenticated, signOut, user } = useAuth();

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const params = new URLSearchParams();
    if (query.trim()) params.set("q", query.trim());
    const search = params.toString();
    navigate({ pathname: "/search", search: search ? `?${search}` : "" });
  }

  async function handleSignOut() {
    setIsSigningOut(true);
    setSignOutError(null);
    try {
      await signOut();
    } catch {
      setSignOutError("MovieMind could not confirm server sign-out. Your local session was cleared.");
    } finally {
      setIsSigningOut(false);
    }
  }

  return (
    <div className="min-h-screen bg-[var(--mm-bg)] text-[var(--mm-text)] lg:grid lg:grid-cols-[15.5rem_minmax(0,1fr)]">
      <a className="sr-only fixed left-4 top-4 z-50 rounded-xl bg-[var(--mm-primary)] px-4 py-3 font-semibold text-white focus:not-sr-only focus:outline-2 focus:outline-offset-2 focus:outline-[var(--mm-accent)]" href="#main-content">Skip to main content</a>
      <Navigation />
      <div className="min-w-0">
        <header className="sticky top-0 z-20 flex min-h-20 flex-wrap items-center justify-between gap-3 border-b border-[var(--mm-border)] bg-[var(--mm-surface-glass)] px-4 py-3 backdrop-blur-xl sm:px-6 lg:px-10">
          <form className="relative order-3 block w-full max-w-none md:order-none md:max-w-xl" onSubmit={submitSearch} role="search">
            <label className="sr-only" htmlFor="global-movie-search">Search for movies or genres</label>
            <Search aria-hidden="true" className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-[var(--mm-text-subtle)]" />
            <Input id="global-movie-search" onChange={(event) => setQuery(event.target.value)} placeholder="Search for movies, genres..." value={query} className="rounded-full pl-12 pr-28" />
            <button className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full px-3 py-2 text-sm font-semibold text-[var(--mm-accent)] transition-colors hover:text-[var(--mm-text)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]" type="submit">Search</button>
          </form>
          <Link className="font-bold tracking-tight text-[var(--mm-text)] md:hidden" to="/">Movie<span className="text-[var(--mm-accent)]">Mind</span></Link>
          <nav aria-label="Account" className="ml-auto flex items-center gap-2">
            <ThemeToggle />
            {isAuthenticated && user ? <><Link aria-label="Open profile" className="flex items-center gap-2 rounded-xl px-2 py-2 text-sm font-semibold text-[var(--mm-text-muted)] transition hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)] sm:px-3" to="/profile"><span className="grid size-7 place-items-center rounded-full bg-[var(--mm-primary-deep)] text-xs text-white">{(user.display_name ?? user.username ?? user.email).slice(0, 1).toUpperCase()}</span><span className="hidden sm:inline">{user.display_name ?? user.username ?? "Account"}</span></Link><Button className="min-h-10 rounded-xl px-3 sm:px-4" disabled={isSigningOut} isLoading={isSigningOut} onClick={() => void handleSignOut()} variant="secondary">Sign out</Button></> : <><Link className="hidden rounded-xl px-3 py-2 text-sm font-semibold text-[var(--mm-text-muted)] transition hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)] sm:block" to="/login">Log in</Link><Link to="/register"><Button className="min-h-10 rounded-xl px-3 sm:px-4"><span className="sm:hidden">Join</span><span className="hidden sm:inline">Create account</span></Button></Link></>}
          </nav>
        </header>
        {signOutError && <p className="border-b border-[var(--mm-border)] bg-[var(--mm-surface)] px-4 py-3 text-center text-sm text-[var(--mm-danger)]" role="alert">{signOutError}</p>}
        <main className="mx-auto w-full max-w-[1440px] px-4 py-6 pb-24 sm:px-6 sm:py-8 lg:px-10 lg:pb-10" id="main-content" tabIndex={-1}><Outlet /></main>
        <footer aria-label="Credits" className="mx-auto flex w-full max-w-[1440px] flex-col items-center gap-3 border-t border-[var(--mm-border)] px-4 py-6 text-center sm:px-6 lg:px-10">
          <a aria-label="The Movie Database website" className="rounded-md bg-white px-2 py-1" href="https://www.themoviedb.org" rel="noreferrer" target="_blank">
            <img alt="The Movie Database" className="h-6 w-auto" src="/tmdb-logo.svg" />
          </a>
          <p className="text-sm font-medium text-[var(--mm-text-muted)]">Credits: This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
        </footer>
      </div>
    </div>
  );
}
