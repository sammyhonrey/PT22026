import { Clapperboard } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";

export function NotFoundPage() {
  return (
    <main className="auth-stage grid min-h-screen place-items-center p-5 text-center text-[var(--mm-text)]">
      <section className="max-w-md rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-10 shadow-[var(--mm-card-shadow)]">
        <Clapperboard aria-hidden="true" className="mx-auto size-12 text-[var(--mm-accent)]" />
        <p className="mt-6 text-sm font-bold tracking-[.2em] text-[var(--mm-accent)]">404</p>
        <h1 className="mt-2 text-3xl font-black">This scene is missing</h1>
        <p className="mt-3 leading-6 text-[var(--mm-text-muted)]">The page you are looking for is not in the MovieMind catalogue.</p>
        <Link className="mt-7 inline-block" to="/"><Button>Back to Home</Button></Link>
      </section>
    </main>
  );
}
