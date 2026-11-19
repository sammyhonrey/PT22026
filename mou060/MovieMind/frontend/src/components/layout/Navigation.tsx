import { Bookmark, Compass, Heart, Home, Settings, Star, UserRound } from "lucide-react";
import { NavLink } from "react-router-dom";

import { useAuth } from "@/features/auth/AuthProvider";

const items = [
  { icon: Home, label: "Home", to: "/" },
  { icon: Compass, label: "Discover", to: "/discover" },
  { icon: Heart, label: "My Taste", to: "/preferences" },
  { icon: Bookmark, label: "My List", to: "/watchlist" },
  { icon: Star, label: "Ratings", to: "/ratings" },
  { icon: UserRound, label: "Profile", to: "/profile" },
  { icon: Settings, label: "Settings", to: "/settings" },
];

export function Navigation() {
  const { isAuthenticated } = useAuth();
  const visibleItems = isAuthenticated ? items : items.slice(0, 2);
  return (
    <>
      <aside className="hidden min-h-screen border-r border-[var(--mm-border)] bg-[var(--mm-bg-elevated)] px-4 py-7 lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col">
        <NavLink aria-label="MovieMind home" className="mb-9 flex justify-center px-2" to="/">
          <img
            alt="MovieMind"
            className="h-auto w-full max-w-[190px] object-contain"
            src="/images/Glossy%20MovieMind%20Cinematic%20Logo.png"
          />
        </NavLink>
        <nav aria-label="Main navigation" className="space-y-1">
          {visibleItems.map(({ icon: Icon, label, to }) => (
            <NavLink
              key={to}
              className={({ isActive }) => `flex min-h-12 items-center gap-3 rounded-2xl px-4 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)] ${isActive ? "bg-linear-to-r from-[var(--mm-primary-deep)] to-[var(--mm-primary)] text-white shadow-[var(--mm-primary-shadow)]" : "text-[var(--mm-text-muted)] hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)]"}`}
              to={to}
            >
              <Icon aria-hidden="true" className="size-5" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-3xl border border-[var(--mm-border)] bg-linear-to-br from-[var(--mm-primary-hover)] to-[var(--mm-surface)] p-5">
          <p className="text-sm font-bold text-[var(--mm-text)]">Build your taste</p>
          <p className="mt-2 text-xs leading-5 text-[var(--mm-text-muted)]">Rate movies to make your recommendations more personal.</p>
        </div>
      </aside>
      <nav aria-label="Mobile navigation" className="fixed inset-x-0 bottom-0 z-30 flex gap-1 overflow-x-auto border-t border-[var(--mm-border)] bg-[var(--mm-surface-glass)] px-2 py-2 backdrop-blur-xl lg:hidden">
        {visibleItems.slice(0, 5).map(({ icon: Icon, label, to }) => (
          <NavLink key={to} className={({ isActive }) => `flex min-h-11 min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-xl px-1 py-1.5 text-[10px] font-semibold focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)] ${isActive ? "bg-[var(--mm-primary)] text-white" : "text-[var(--mm-text-muted)]"}`} to={to}>
            <Icon aria-hidden="true" className="size-5" />
            {label}
          </NavLink>
        ))}
      </nav>
    </>
  );
}
