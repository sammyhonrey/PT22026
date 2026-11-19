import { ArrowRight, Sparkles, Star } from "lucide-react";
import { Link } from "react-router-dom";

import { MoviePoster } from "@/components/movies/MoviePoster";
import { Button } from "@/components/ui/Button";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { useAuth } from "@/features/auth/AuthProvider";
import { useMyPreferences } from "@/features/preferences/hooks";
import { useMyRecommendations } from "@/features/recommendations/hooks";
import { RecommendationPreviewSkeleton } from "@/features/recommendations/components/RecommendationSkeletons";
import { isApiClientError } from "@/lib/api/errors";

export function RecommendationPreview() {
  const { isAuthenticated } = useAuth();
  const preferences = useMyPreferences(isAuthenticated);
  const recommendations = useMyRecommendations(3, isAuthenticated);
  if (!isAuthenticated) return <GuestPreview />;
  if (recommendations.isFetching || preferences.isFetching) return <RecommendationPreviewSkeleton />;
  if (recommendations.error) return <aside className="space-y-4 rounded-3xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6 shadow-[var(--mm-card-shadow)]"><StatusPanel tone="error">{isApiClientError(recommendations.error) ? recommendations.error.message : "We could not load your recommendations."}</StatusPanel><Button isLoading={recommendations.isFetching} onClick={() => void recommendations.refetch()} variant="secondary">Try again</Button></aside>;
  const items = recommendations.data?.items.slice(0, 3) ?? [];
  const hasSavedPreferences = Boolean(preferences.data && (preferences.data.favorite_genres.length || preferences.data.favorite_movie_ids.length || preferences.data.description));
  if (!items.length) return <aside className="space-y-4 rounded-3xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6 shadow-[var(--mm-card-shadow)]">{preferences.error && <div className="space-y-3"><StatusPanel tone="error">We could not load your taste profile.</StatusPanel><Button isLoading={preferences.isFetching} onClick={() => void preferences.refetch()} variant="secondary">Retry taste profile</Button></div>}<Sparkles aria-hidden="true" className="size-7 text-[var(--mm-accent)]" /><h2 className="mt-5 text-xl font-bold">Recommendations are waiting</h2><p className="mt-3 text-sm leading-6 text-[var(--mm-text-muted)]">{hasSavedPreferences ? "Your saved taste profile is ready for the next run." : "Rate movies to help MovieMind find more to show you."}</p><Link className="mt-6 inline-flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)] hover:text-[var(--mm-text)]" to={hasSavedPreferences ? "/preferences" : "/discover"}>{hasSavedPreferences ? "Refine my taste" : "Explore movies"} <ArrowRight aria-hidden="true" className="size-4" /></Link></aside>;
  return (
    <aside aria-busy={recommendations.isFetching} aria-labelledby="home-recommendations-title" className="recommendation-panel rounded-3xl border p-5 shadow-[var(--mm-card-shadow)] sm:p-6">
      <div className="flex items-center gap-2 text-[var(--mm-accent)]">
        <Sparkles aria-hidden="true" className="size-5" />
        <p className="text-xs font-bold uppercase tracking-[0.15em]">For your next movie night</p>
      </div>
      <h2 className="mt-3 text-xl font-black" id="home-recommendations-title">Your Recommendations</h2>
      <p className="mt-2 text-sm leading-5 text-[var(--mm-text-muted)]">
        {recommendations.data?.is_fallback
          ? "Popular picks while you get started"
          : hasSavedPreferences
            ? "Personalized from your taste profile"
            : "Picked for you from the MovieMind catalogue"}
      </p>
      {preferences.error && <div className="mt-4 space-y-3"><StatusPanel tone="error">We could not load your saved taste profile. Recommendations are still available.</StatusPanel><Button isLoading={preferences.isFetching} onClick={() => void preferences.refetch()} variant="secondary">Retry taste profile</Button></div>}
      <div className="mt-5 space-y-3">
        {items.map(({ movie, reason }, index) => (
          <article className="mm-content-enter flex min-w-0 gap-3 rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-subtle-surface)] p-2.5" key={movie.id}>
            <Link aria-label={`View details for ${movie.title}`} className="w-16 shrink-0 overflow-hidden rounded-xl focus-visible:outline-2 focus-visible:outline-[var(--mm-accent)]" to={`/movies/${movie.id}`}>
              <MoviePoster className="aspect-[2/3]" movie={movie} priority={index === 0} />
            </Link>
            <div className="min-w-0 py-1">
              <Link className="line-clamp-2 text-sm font-bold leading-5 hover:text-[var(--mm-accent)]" to={`/movies/${movie.id}`}>{movie.title}</Link>
              {movie.average_rating !== null && <p className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-[var(--mm-gold)]"><Star aria-hidden="true" className="size-3 fill-current" /> {movie.average_rating.toFixed(1)}</p>}
              <p className="mt-1 line-clamp-2 text-xs leading-4 text-[var(--mm-text-muted)]">{reason}</p>
            </div>
          </article>
        ))}
      </div>
      <Link className="mt-5 inline-flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)] transition hover:text-[var(--mm-text)]" to="/recommendations">
        See all recommendations
        <ArrowRight aria-hidden="true" className="size-4" />
      </Link>
    </aside>
  );
}

function GuestPreview() {
  return <aside className="rounded-3xl border border-[var(--mm-border)] bg-linear-to-br from-[var(--mm-primary-hover)] to-[var(--mm-surface)] p-6 shadow-[var(--mm-card-shadow)]"><Sparkles aria-hidden="true" className="size-7 text-[var(--mm-accent)]" /><h2 className="mt-6 text-xl font-bold">Recommendations that grow with you</h2><p className="mt-3 text-sm leading-6 text-[var(--mm-text-muted)]">Create an account and rate movies to get recommendations based on your taste.</p><Link className="mt-6 inline-flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)] hover:text-[var(--mm-text)]" to="/register">Create an account <ArrowRight aria-hidden="true" className="size-4" /></Link></aside>;
}
