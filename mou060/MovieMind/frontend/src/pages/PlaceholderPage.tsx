import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";

type PlaceholderPageProps = { description: string; title: string };

export function PlaceholderPage({ description, title }: PlaceholderPageProps) {
  return (
    <div className="mx-auto max-w-3xl pt-8 sm:pt-14">
      <p className="text-sm font-semibold text-[var(--mm-accent)]">MovieMind foundation</p>
      <h1 className="mt-2 text-4xl font-black tracking-tight text-[var(--mm-text)]">{title}</h1>
      <div className="mt-8"><EmptyState title="Coming in a dedicated feature phase" description={description} action={<Link to="/"><Button variant="secondary">Back to Home</Button></Link>} /></div>
    </div>
  );
}
