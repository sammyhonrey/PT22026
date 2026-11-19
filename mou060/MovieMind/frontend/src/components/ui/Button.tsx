import { type ButtonHTMLAttributes, type PropsWithChildren } from "react";

type ButtonVariant = "primary" | "secondary" | "ghost" | "icon";

type ButtonProps = PropsWithChildren<
  ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    isLoading?: boolean;
  }
>;

const variants: Record<ButtonVariant, string> = {
  primary: "bg-[var(--mm-primary)] text-white shadow-[var(--mm-primary-shadow)] hover:bg-[var(--mm-primary-hover)]",
  secondary: "border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] text-[var(--mm-text)] hover:bg-[var(--mm-surface-hover)]",
  ghost: "text-[var(--mm-text-muted)] hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)]",
  icon: "size-11 rounded-full border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] p-0 text-[var(--mm-text)] hover:bg-[var(--mm-hover-surface-strong)]",
};

export function Button({ children, className = "", disabled, isLoading, type = "button", variant = "primary", ...props }: ButtonProps) {
  return (
    <button
      aria-busy={isLoading || undefined}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-2xl px-5 text-sm font-semibold transition-[color,background-color,border-color,opacity,transform,filter] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)] motion-safe:enabled:active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60 ${variants[variant]} ${className}`}
      disabled={disabled || isLoading}
      type={type}
      {...props}
    >
      {isLoading && <span aria-hidden="true" className="size-4 rounded-full border-2 border-current border-r-transparent motion-safe:animate-spin" />}
      {children}
    </button>
  );
}
