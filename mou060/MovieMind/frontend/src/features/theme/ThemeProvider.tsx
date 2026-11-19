import { createContext, useCallback, useContext, useEffect, useMemo, useState, type PropsWithChildren } from "react";

export type Theme = "dark" | "light";

const storageKey = "moviemind-theme";
const systemColorScheme = "(prefers-color-scheme: light)";

type ThemeContextValue = {
  theme: Theme;
  toggleTheme: () => void;
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

function isTheme(value: string | null | undefined): value is Theme {
  return value === "dark" || value === "light";
}

function readSavedTheme() {
  try {
    const savedTheme = window.localStorage.getItem(storageKey);
    return isTheme(savedTheme) ? savedTheme : null;
  } catch (error) {
    console.warn("MovieMind could not read the saved theme preference.", error);
    return null;
  }
}

function getSystemTheme(): Theme {
  return typeof window.matchMedia === "function" && window.matchMedia(systemColorScheme).matches
    ? "light"
    : "dark";
}

function getInitialTheme() {
  const savedTheme = readSavedTheme();
  const documentTheme = document.documentElement.dataset.theme;
  return {
    hasSavedPreference: savedTheme !== null,
    theme: savedTheme ?? (isTheme(documentTheme) ? documentTheme : getSystemTheme()),
  };
}

export function ThemeProvider({ children }: PropsWithChildren) {
  const [initialTheme] = useState(getInitialTheme);
  const [theme, setTheme] = useState<Theme>(initialTheme.theme);
  const [hasSavedPreference, setHasSavedPreference] = useState(initialTheme.hasSavedPreference);

  const selectTheme = useCallback((nextTheme: Theme) => {
    setTheme(nextTheme);
    setHasSavedPreference(true);
    try {
      window.localStorage.setItem(storageKey, nextTheme);
    } catch (error) {
      console.warn("MovieMind could not save the theme preference.", error);
    }
  }, []);

  const toggleTheme = useCallback(() => {
    selectTheme(theme === "dark" ? "light" : "dark");
  }, [selectTheme, theme]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      "content",
      theme === "light" ? "#F5F5F5" : "#141414",
    );
  }, [theme]);

  useEffect(() => {
    if (hasSavedPreference) return;
    if (typeof window.matchMedia !== "function") return;
    const mediaQuery = window.matchMedia(systemColorScheme);
    const handleColorSchemeChange = (event: MediaQueryListEvent) => {
      setTheme(event.matches ? "light" : "dark");
    };
    mediaQuery.addEventListener("change", handleColorSchemeChange);
    return () => mediaQuery.removeEventListener("change", handleColorSchemeChange);
  }, [hasSavedPreference]);

  useEffect(() => {
    function handleStorageChange(event: StorageEvent) {
      if (event.key !== storageKey && event.key !== null) return;
      const savedTheme = isTheme(event.newValue) ? event.newValue : null;
      setHasSavedPreference(savedTheme !== null);
      setTheme(savedTheme ?? getSystemTheme());
    }

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within a ThemeProvider.");
  return context;
}
