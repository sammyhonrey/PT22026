import { Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { MoviePoster } from "@/components/movies/MoviePoster";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { Pagination } from "@/features/discovery/components/Pagination";
import { useDeleteRating, useMyRatings } from "@/features/ratings/hooks";
import { isApiClientError } from "@/lib/api/errors";

const pageSize = 20;

export function RatingsPage() {
  const [offset, setOffset] = useState(0);
  const ratings = useMyRatings(offset, pageSize);
  const remove = useDeleteRating();
  const [message, setMessage] = useState<string | null>(null);

  async function deleteItem(movieId: string) {
    setMessage(null);
    try { await remove.mutateAsync(movieId); setMessage("Rating removed."); } catch (error) { setMessage(isApiClientError(error) ? error.message : "We could not remove your rating."); }
  }

  return <div className="space-y-7"><header><p className="flex items-center gap-2 text-sm font-bold text-[var(--mm-gold)]"><Star aria-hidden="true" className="size-4 fill-current" /> Personal ratings</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">My Ratings</h1><p className="mt-2 text-[var(--mm-text-muted)]">Your ratings help MovieMind improve your recommendations.</p></header>{message && <StatusPanel tone={message.includes("could not") ? "error" : "success"}>{message}</StatusPanel>}{ratings.isLoading ? <div aria-busy="true" aria-label="Loading your ratings" className="space-y-3" role="status">{Array.from({ length: 5 }, (_, index) => <article className="flex gap-4 rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-3" key={index}><Skeleton className="aspect-[2/3] w-18 shrink-0 rounded-xl" /><div className="flex-1 space-y-3 py-2"><Skeleton className="h-5 w-3/5" /><Skeleton className="h-4 w-1/4" /><Skeleton className="h-4 w-1/3" /></div></article>)}</div> : <>{ratings.isFetching && <StatusPanel>Refreshing your ratings…</StatusPanel>}{ratings.error ? <div className="space-y-4"><StatusPanel tone="error">{isApiClientError(ratings.error) ? ratings.error.message : "We could not load your ratings."}</StatusPanel><Button isLoading={ratings.isFetching} onClick={() => void ratings.refetch()} variant="secondary">Try again</Button></div> : !ratings.data?.items.length ? <EmptyState title="You have not rated any movies yet" description="Rate a few movies to make your recommendations more personal." action={<Link to="/discover"><Button>Discover Movies</Button></Link>} /> : <><div className="space-y-3">{ratings.data.items.map((rating) => <article className="mm-content-enter flex gap-4 rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-3 shadow-[var(--mm-card-shadow)]" key={rating.id}><Link aria-label={`View ${rating.movie.title}`} className="shrink-0 focus-visible:outline-2 focus-visible:outline-[var(--mm-accent)]" to={`/movies/${rating.movie.id}`}><MoviePoster className="aspect-[2/3] w-18 rounded-xl" movie={rating.movie} /></Link><div className="min-w-0 flex-1 py-1"><Link className="text-lg font-bold hover:text-[var(--mm-accent)]" to={`/movies/${rating.movie.id}`}>{rating.movie.title}</Link><p className="mt-1 text-sm text-[var(--mm-text-muted)]">{rating.movie.release_year ?? "Year unavailable"}</p><p className="mt-3 inline-flex items-center gap-1 font-bold text-[var(--mm-gold)]"><Star aria-hidden="true" className="size-4 fill-current" /> Your rating: {rating.score.toFixed(1)}</p></div><Button aria-label={`Remove rating for ${rating.movie.title}`} className="min-h-10 shrink-0 rounded-xl px-3" disabled={remove.isPending} isLoading={remove.isPending && remove.variables === rating.movie_id} onClick={() => void deleteItem(rating.movie_id)} variant="ghost"><Trash2 aria-hidden="true" className="size-4" /></Button></article>)}</div><Pagination limit={ratings.data.limit} offset={ratings.data.offset} onPageChange={setOffset} total={ratings.data.total} /></>}</>}</div>;
}
