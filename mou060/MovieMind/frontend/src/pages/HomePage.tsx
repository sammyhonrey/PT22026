import { ArrowRight, Compass, Flame, Sparkles, Star } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";

import { MovieCard } from "@/components/movies/MovieCard";
import { MovieCardSkeleton } from "@/components/movies/MovieCardSkeleton";
import { Button } from "@/components/ui/Button";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { usePopularMovies, useMovies } from "@/features/discovery/hooks";
import { RecommendationPreview } from "@/features/recommendations/components/RecommendationPreview";
import { isApiClientError } from "@/lib/api/errors";
import type { Movie } from "@/types/api";

export function HomePage() {
  const trending = usePopularMovies(8);
  const topRated = useMovies({ direction: "desc", limit: 8, sort: "average_rating" });

  return (
    <div className="space-y-8 pb-4 lg:space-y-10">
      <section className="home-hero relative isolate flex min-h-[27rem] items-center overflow-hidden rounded-[28px] border border-[var(--mm-border)] px-6 py-10 shadow-[var(--mm-card-shadow)] sm:min-h-[30rem] sm:px-10 sm:py-14 lg:min-h-[31rem] lg:px-14">
        <div className="home-hero-glow pointer-events-none absolute inset-0" />
        <div className="relative z-10 max-w-xl">
          <p className="mb-4 flex items-center gap-2 text-sm font-bold uppercase tracking-[0.18em] text-[var(--mm-accent)]">
            <Sparkles aria-hidden="true" className="size-4" />
            MovieMind
          </p>
          <h1 className="text-4xl font-black leading-[1.04] tracking-tight text-[var(--mm-text)] sm:text-5xl lg:text-6xl">
            Your Personal <span className="text-gradient">Movie Companion</span>
          </h1>
          <p className="mt-5 max-w-lg text-base leading-7 text-[var(--mm-text-muted)] sm:text-lg">
            Get AI-powered recommendations and discover movies you&apos;ll love!
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-[var(--mm-primary)] px-6 text-sm font-semibold text-white shadow-[var(--mm-primary-shadow)] transition-colors hover:bg-[var(--mm-primary-hover)]" to="/discover">
              <Compass aria-hidden="true" className="size-4" />
              Start Exploring
            </Link>
            <Link className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-6 text-sm font-semibold text-[var(--mm-text)] transition hover:bg-[var(--mm-surface-hover)]" to="/preferences">
              Learn More
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          </div>
        </div>
      </section>

      <div className="grid items-start gap-8 xl:grid-cols-[minmax(0,1fr)_21rem] xl:gap-9">
        <div className="min-w-0 space-y-9">
          <HomeMovieRail
            icon={Flame}
            movies={trending.data}
            error={trending.error}
            isLoading={trending.isFetching}
            onRetry={() => void trending.refetch()}
            seeAllTo="/discover?sort=ratings_count&direction=desc"
            title="Trending Now"
          />
          <HomeMovieRail
            icon={Star}
            movies={topRated.data?.items}
            error={topRated.error}
            isLoading={topRated.isFetching}
            onRetry={() => void topRated.refetch()}
            seeAllTo="/discover?sort=average_rating&direction=desc"
            title="Top Rated Movies"
          />
          <section className="home-feature-panel relative overflow-hidden rounded-3xl border p-6 shadow-[var(--mm-card-shadow)] sm:p-8">
            <div aria-hidden="true" className="pointer-events-none absolute -right-10 -top-20 size-64 rounded-full border border-[var(--mm-decoration-border)]" />
            <div className="relative z-10 max-w-xl">
              <p className="text-xs font-bold uppercase tracking-[0.16em] text-[var(--mm-accent)]">Your next favorite is waiting</p>
              <h2 className="mt-3 text-2xl font-black tracking-tight sm:text-3xl">Ready for your next movie adventure?</h2>
              <p className="mt-3 max-w-lg text-sm leading-6 text-[var(--mm-text-muted)]">Explore the catalogue and find a story worth settling in for.</p>
              <Link className="mt-6 inline-flex min-h-11 items-center gap-2 rounded-xl bg-[var(--mm-primary)] px-5 text-sm font-semibold text-white shadow-[var(--mm-primary-shadow)] transition-colors hover:bg-[var(--mm-primary-hover)]" to="/discover">
                Discover Movies
                <ArrowRight aria-hidden="true" className="size-4" />
              </Link>
            </div>
          </section>
        </div>
        <div className="min-w-0 xl:sticky xl:top-24">
          <RecommendationPreview />
        </div>
      </div>
    </div>
  );
}

type HomeMovieRailProps = {
  error: unknown;
  icon: LucideIcon;
  isLoading: boolean;
  movies: Movie[] | undefined;
  onRetry: () => void;
  seeAllTo: string;
  title: string;
};

function HomeMovieRail({
  error,
  icon: Icon,
  isLoading,
  movies,
  onRetry,
  seeAllTo,
  title,
}: HomeMovieRailProps) {
  return (
    <section aria-busy={isLoading} aria-label={title} className="min-w-0">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h2 className="flex items-center gap-2 text-xl font-black tracking-tight sm:text-2xl">
          <Icon aria-hidden="true" className="size-5 text-[var(--mm-accent)]" />
          {title}
        </h2>
        <Link className="inline-flex shrink-0 items-center gap-1 text-sm font-semibold text-[var(--mm-accent)] transition hover:text-[var(--mm-text)]" to={seeAllTo}>
          See All
          <ArrowRight aria-hidden="true" className="size-4" />
        </Link>
      </div>
      {isLoading ? (
        <div aria-busy="true" aria-label={`Loading ${title}`} className="flex gap-4 overflow-hidden" role="status">
          {Array.from({ length: 4 }, (_, index) => <MovieCardSkeleton className="w-40 shrink-0 sm:w-48" key={index} />)}
        </div>
      ) : error ? (
        <div className="space-y-3">
          <StatusPanel tone="error">{isApiClientError(error) ? error.message : "We could not load movies. Please try again."}</StatusPanel>
          <Button onClick={onRetry} variant="secondary">Try again</Button>
        </div>
      ) : movies?.length ? (
        <div aria-label={`${title} movies`} className="-mx-1 flex snap-x snap-mandatory gap-4 overflow-x-auto px-1 pb-4">
          {movies.map((movie, index) => (
            <MovieCard
              className="w-40 shrink-0 snap-start sm:w-48"
              key={movie.id}
              movie={movie}
              priority={index < 3}
              showWatchlistButton
            />
          ))}
        </div>
      ) : (
        <StatusPanel>No movies are available in this section yet.</StatusPanel>
      )}
    </section>
  );
}
