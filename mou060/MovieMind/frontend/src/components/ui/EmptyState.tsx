import { Film } from "lucide-react";
import { type ReactNode } from "react";

type EmptyStateProps = {
  action?: ReactNode;
  description: string;
  title: string;
};

export function EmptyState({ action, description, title }: EmptyStateProps) {
  return (
    <section className="rounded-3xl border border-dashed border-[var(--mm-border)] bg-[var(--mm-surface)] px-6 py-12 text-center shadow-[var(--mm-card-shadow)]">
      <Film aria-hidden="true" className="mx-auto mb-4 size-9 text-[var(--mm-accent)]" />
      <h2 className="text-xl font-bold text-[var(--mm-text)]">{title}</h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[var(--mm-text-muted)]">{description}</p>
      {action && <div className="mt-6">{action}</div>}
    </section>
  );
}
