import { ArrowLeft, CalendarDays, Film, Star, Tags } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { MoviePoster } from "@/components/movies/MoviePoster";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { useMovie } from "@/features/discovery/hooks";
import { RatingControl } from "@/features/ratings/components/RatingControl";
import { WatchlistButton } from "@/features/watchlist/components/WatchlistButton";
import { isApiClientError } from "@/lib/api/errors";

function isUuid(value: string | undefined) {
  return Boolean(value && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value));
}

export function MovieDetailsPage() {
  const { movieId } = useParams();
  const validId = isUuid(movieId);
  const movie = useMovie(movieId ?? "", validId);

  if (!validId) return <NotFoundMovie />;
  if (movie.isFetching) return <MovieDetailsSkeleton />;
  if (movie.error) {
    if (isApiClientError(movie.error) && movie.error.status === 404) return <NotFoundMovie />;
    const message = isApiClientError(movie.error) ? movie.error.message : "We could not load this movie. Please try again.";
    return <div className="space-y-4"><StatusPanel tone="error">{message}</StatusPanel><Button onClick={() => void movie.refetch()} variant="secondary">Try again</Button></div>;
  }
  if (!movie.data) return <NotFoundMovie />;

  const item = movie.data;
  return (
    <article className="mx-auto max-w-5xl">
      <Link className="mb-6 inline-flex items-center gap-2 text-sm font-semibold text-[var(--mm-text-muted)] hover:text-[var(--mm-text)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]" to="/discover"><ArrowLeft aria-hidden="true" className="size-4" /> Back to Discover</Link>
      <div className="overflow-hidden rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] shadow-[var(--mm-card-shadow)]">
        <div className="movie-detail-backdrop relative overflow-hidden p-6 sm:p-10">
          <div className="relative grid gap-7 sm:grid-cols-[13rem_minmax(0,1fr)] sm:items-end">
            <MoviePoster className="mx-auto aspect-[2/3] w-44 rounded-2xl border border-[var(--mm-border)] shadow-2xl sm:mx-0 sm:w-full" movie={item} priority />
            <div>
              <p className="text-sm font-bold text-[var(--mm-accent)]">Movie details</p>
              <h1 className="mt-2 text-3xl font-black tracking-tight sm:text-5xl">{item.title}</h1>
              <div className="mt-5 flex flex-wrap gap-3 text-sm text-[var(--mm-text-muted)]">
                {item.release_year !== null && <span className="inline-flex items-center gap-1.5"><CalendarDays aria-hidden="true" className="size-4 text-[var(--mm-accent)]" /> {item.release_year}</span>}
                {item.average_rating !== null && <span className="inline-flex items-center gap-1.5 font-semibold text-[var(--mm-gold)]"><Star aria-hidden="true" className="size-4 fill-current" /> {item.average_rating.toFixed(1)} <span className="font-normal text-[var(--mm-text-muted)]">average rating</span></span>}
                {item.ratings_count !== null && <span>{item.ratings_count} {item.ratings_count === 1 ? "rating" : "ratings"}</span>}
              </div>
              {item.genres.length > 0 && <div className="mt-5 flex flex-wrap gap-2">{item.genres.map((genre) => <span className="inline-flex items-center gap-1 rounded-full border border-[var(--mm-border)] bg-[var(--mm-subtle-surface)] px-3 py-1.5 text-xs font-semibold text-[var(--mm-text-muted)]" key={genre}><Tags aria-hidden="true" className="size-3" /> {genre}</span>)}</div>}
            </div>
          </div>
        </div>
        <div className="space-y-8 p-6 sm:p-10"><div><h2 className="text-xl font-bold">Overview</h2><p className="mt-3 max-w-3xl whitespace-pre-line leading-7 text-[var(--mm-text-muted)]">{item.overview ?? "An overview is not available for this movie."}</p></div><div className="flex flex-wrap items-start gap-4"><WatchlistButton movieId={item.id} /></div><RatingControl movieId={item.id} /><div className="rounded-2xl border border-dashed border-[var(--mm-border)] bg-[var(--mm-bg-elevated)] p-4 text-sm text-[var(--mm-text-muted)]"><Film aria-hidden="true" className="mr-2 inline size-4 text-[var(--mm-accent)]" /> Your rating and saved status are confirmed by MovieMind before this page is updated.</div></div>
      </div>
    </article>
  );
}

function NotFoundMovie() {
  return <div className="mx-auto max-w-2xl pt-10"><EmptyState title="This movie is missing" description="The movie may have been removed, or the link is not valid." action={<Link to="/discover"><Button>Browse movies</Button></Link>} /></div>;
}

function MovieDetailsSkeleton() {
  return <div aria-busy="true" aria-label="Loading movie details" className="mx-auto max-w-5xl overflow-hidden rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)]" role="status"><div className="grid gap-7 p-6 sm:grid-cols-[13rem_minmax(0,1fr)] sm:p-10"><Skeleton className="aspect-[2/3] rounded-2xl" /><div className="space-y-4 self-end"><Skeleton className="h-4 w-24" /><Skeleton className="h-12 w-4/5" /><Skeleton className="h-5 w-1/2" /></div></div><div className="space-y-3 border-t border-[var(--mm-border)] p-6 sm:p-10"><Skeleton className="h-6 w-28" /><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-4/5" /></div></div>;
}
