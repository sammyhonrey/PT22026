import { Star, Trash2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/Button";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { useDeleteRating, useMyRating, useUpsertRating } from "@/features/ratings/hooks";
import { useAuth } from "@/features/auth/AuthProvider";
import { isApiClientError } from "@/lib/api/errors";

const scores = Array.from({ length: 10 }, (_, index) => (index + 1) / 2);

export function RatingControl({ movieId }: { movieId: string }) {
  const { isAuthenticated } = useAuth();
  const current = useMyRating(movieId, isAuthenticated);
  const save = useUpsertRating();
  const remove = useDeleteRating();
  const [message, setMessage] = useState<string | null>(null);
  const pending = save.isPending || remove.isPending;

  if (!isAuthenticated) return <div className="rounded-2xl border border-dashed border-[var(--mm-border)] bg-[var(--mm-bg-elevated)] p-4 text-sm text-[var(--mm-text-muted)]">Sign in to rate this movie.</div>;
  if (current.isLoading) return <StatusPanel>Loading your rating…</StatusPanel>;
  if (current.error) return <div className="space-y-3"><StatusPanel tone="error">{isApiClientError(current.error) ? current.error.message : "We could not load your rating."}</StatusPanel><Button isLoading={current.isFetching} onClick={() => void current.refetch()} variant="secondary">Try again</Button></div>;

  async function selectScore(score: number) {
    setMessage(null);
    try {
      await save.mutateAsync({ movie_id: movieId, score });
      setMessage(`Your rating of ${score.toFixed(1)} was saved.`);
    } catch (error) {
      setMessage(isApiClientError(error) ? error.message : "We could not save your rating.");
    }
  }

  async function clearRating() {
    setMessage(null);
    try {
      await remove.mutateAsync(movieId);
      setMessage("Your rating was removed.");
    } catch (error) {
      setMessage(isApiClientError(error) ? error.message : "We could not remove your rating.");
    }
  }

  return (
    <section aria-labelledby="your-rating" className="rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-bg-elevated)] p-4">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="font-bold" id="your-rating">Your rating</h2><p className="mt-1 text-sm text-[var(--mm-text-muted)]">Choose a score from 0.5 to 5.0.</p></div>{current.data && <Button className="min-h-9 rounded-xl px-3" isLoading={remove.isPending} onClick={() => void clearRating()} variant="ghost"><Trash2 aria-hidden="true" className="size-4" /> Remove</Button>}</div>
      <div aria-busy={pending} aria-label="Choose your rating" className="mt-4 grid grid-cols-5 gap-2 sm:grid-cols-10" role="radiogroup">
        {scores.map((score) => {
          const selected = current.data?.score === score;
          return <button aria-checked={selected} aria-label={`Rate ${score.toFixed(1)} out of 5`} className={`flex min-h-11 flex-col items-center justify-center rounded-xl border text-xs font-bold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)] ${selected ? "border-[var(--mm-primary)] bg-[var(--mm-primary)] text-white" : "border-[var(--mm-border)] bg-[var(--mm-surface)] text-[var(--mm-text-muted)] hover:border-[var(--mm-accent)] hover:text-[var(--mm-gold)]"}`} disabled={pending} key={score} onClick={() => void selectScore(score)} role="radio" type="button"><Star aria-hidden="true" className={`size-3.5 ${selected ? "fill-current" : ""}`} />{score.toFixed(1)}</button>;
        })}
      </div>
      {pending && <p className="mt-4 text-sm text-[var(--mm-text-muted)]" role="status">{save.isPending ? "Saving your rating…" : "Removing your rating…"}</p>}
      {message && <p className={`mt-4 text-sm ${message.includes("could not") || message.includes("Too many") ? "text-[var(--mm-danger)]" : "text-[var(--mm-success)]"}`} role="status">{message}</p>}
    </section>
  );
}
