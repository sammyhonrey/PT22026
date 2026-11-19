import { Moon, Sun } from "lucide-react";

import { useTheme } from "@/features/theme/ThemeProvider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <button
      aria-checked={theme === "light"}
      aria-label="Light mode"
      className="group relative grid size-11 shrink-0 place-items-center rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] text-[var(--mm-text)] transition-colors hover:border-[var(--mm-primary)] hover:bg-[var(--mm-surface-hover)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]"
      onClick={toggleTheme}
      role="switch"
      title={`Switch to ${nextTheme} mode`}
      type="button"
    >
      {theme === "light" ? <Sun aria-hidden="true" className="size-5" /> : <Moon aria-hidden="true" className="size-5" />}
      <span className="pointer-events-none absolute right-0 top-full z-50 mt-2 hidden whitespace-nowrap rounded-lg border border-[var(--mm-border)] bg-[var(--mm-surface)] px-3 py-2 text-xs font-medium text-[var(--mm-text)] shadow-[var(--mm-card-shadow)] group-hover:block group-focus-visible:block">
        Switch to {nextTheme} mode
      </span>
    </button>
  );
}
