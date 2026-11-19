import { Film } from "lucide-react";
import { useState } from "react";

import { getMoviePosterUrl } from "@/lib/movie-posters";
import { Skeleton } from "@/components/ui/Skeleton";
import type { Movie } from "@/types/api";

type MoviePosterProps = {
  className?: string;
  movie: Pick<Movie, "movielens_id" | "title">;
  priority?: boolean;
};

export function MoviePoster({ className = "", movie, priority = false }: MoviePosterProps) {
  const posterUrl = getMoviePosterUrl(movie.movielens_id);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const canShowImage = posterUrl !== null && posterUrl !== failedUrl;

  return (
    <div className={`relative overflow-hidden bg-[var(--mm-bg-elevated)] ${className}`}>
      {canShowImage ? (
        <>
          {loadedUrl !== posterUrl && <Skeleton className="absolute inset-0 rounded-none" />}
          <img
            alt={`Poster for ${movie.title}`}
            className={`relative size-full object-cover transition-opacity duration-300 ${loadedUrl === posterUrl ? "opacity-100" : "opacity-0"}`}
            decoding="async"
            loading={priority ? "eager" : "lazy"}
            onError={() => setFailedUrl(posterUrl)}
            onLoad={() => setLoadedUrl(posterUrl)}
            src={posterUrl}
          />
        </>
      ) : (
        <div className="movie-poster-fallback flex size-full flex-col items-center justify-center gap-3 px-4 text-center">
          <Film aria-hidden="true" className="size-9 text-[var(--mm-accent)]" />
          <span className="line-clamp-2 text-xs font-semibold text-[var(--mm-text-muted)]">{movie.title}</span>
        </div>
      )}
    </div>
  );
}
