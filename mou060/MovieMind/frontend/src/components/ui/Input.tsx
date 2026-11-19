import { forwardRef, type InputHTMLAttributes } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({ className = "", ...props }, ref) {
  return (
    <input
      className={`min-h-11 w-full rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-4 text-[var(--mm-text)] outline-none transition-[border-color,box-shadow,opacity] duration-200 placeholder:text-[var(--mm-text-subtle)] focus:border-[var(--mm-accent)] focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)]/70 disabled:cursor-not-allowed disabled:opacity-60 ${className}`}
      ref={ref}
      {...props}
    />
  );
});
