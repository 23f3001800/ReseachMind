import { useCallback, useEffect, useState } from "react";

/** State mirrored into localStorage, tolerant of private-mode write failures. */
export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* quota or private mode — in-memory state still works */
    }
  }, [key, value]);

  return [value, setValue] as const;
}

export type ThemeMode = "light" | "dark" | "system";

/** Theme with a system option that keeps tracking the OS after selection. */
export function useTheme() {
  const [mode, setMode] = useState<ThemeMode>(() => {
    try {
      return (localStorage.getItem("ra.theme") as ThemeMode) || "system";
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const resolved = mode === "system" ? (mql.matches ? "dark" : "light") : mode;
      document.documentElement.setAttribute("data-theme", resolved);
    };
    apply();
    try {
      localStorage.setItem("ra.theme", mode);
    } catch {
      /* ignore */
    }
    if (mode === "system") {
      mql.addEventListener("change", apply);
      return () => mql.removeEventListener("change", apply);
    }
    return undefined;
  }, [mode]);

  const cycle = useCallback(() => {
    setMode((m) => (m === "light" ? "dark" : m === "dark" ? "system" : "light"));
  }, []);

  return { mode, setMode, cycle };
}
