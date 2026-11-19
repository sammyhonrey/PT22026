import { LoaderCircle } from "lucide-react";

export function PageLoader({ label }: { label: string }) {
  return (
    <div aria-busy="true" aria-label={label} className="mx-auto flex max-w-3xl items-center gap-3 pt-10 text-sm text-[var(--mm-text-muted)]" role="status">
      <LoaderCircle aria-hidden="true" className="size-5 shrink-0 text-[var(--mm-accent)] motion-safe:animate-spin" />
      {label}
    </div>
  );
}
