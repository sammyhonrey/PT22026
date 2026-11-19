import { ArrowRight, Sparkles, Star } from "lucide-react";
import { Link } from "react-router-dom";

import { MoviePoster } from "@/components/movies/MoviePoster";
import type { Recommendation } from "@/types/api";

export function RecommendationCard({ recommendation, featured = false }: { recommendation: Recommendation; featured?: boolean }) {
  const { movie } = recommendation;
  return (
    <article className={`mm-content-enter overflow-hidden rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] shadow-[var(--mm-card-shadow)] transition-[transform,border-color] duration-200 motion-safe:hover:-translate-y-0.5 motion-safe:hover:border-[var(--mm-primary)] ${featured ? "md:grid md:grid-cols-[11rem_minmax(0,1fr)]" : ""}`}>
      <Link aria-label={`View details for ${movie.title}`} className="block focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]" to={`/movies/${movie.id}`}><MoviePoster className={featured ? "aspect-[2/3] h-full min-h-64" : "aspect-[2/3]"} movie={movie} priority={featured} /></Link>
      <div className={`p-4 ${featured ? "flex flex-col justify-center sm:p-6" : ""}`}><p className="inline-flex items-center gap-1 text-xs font-bold text-[var(--mm-accent)]"><Sparkles aria-hidden="true" className="size-3.5" /> {recommendation.source}</p><Link className="mt-2 block text-lg font-bold hover:text-[var(--mm-accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)]" to={`/movies/${movie.id}`}>{movie.title}</Link>{movie.average_rating !== null && <p className="mt-2 inline-flex items-center gap-1 text-sm font-semibold text-[var(--mm-gold)]"><Star aria-hidden="true" className="size-3.5 fill-current" /> {movie.average_rating.toFixed(1)} <span className="font-normal text-[var(--mm-text-muted)]">average rating</span></p>}<p className="mt-3 text-sm leading-6 text-[var(--mm-text-muted)]">{recommendation.reason}</p>{featured && <Link className="mt-5 inline-flex items-center gap-1 text-sm font-bold text-[var(--mm-accent)] hover:text-[var(--mm-text)]" to={`/movies/${movie.id}`}>View movie <ArrowRight aria-hidden="true" className="size-4" /></Link>}</div>
    </article>
  );
}
