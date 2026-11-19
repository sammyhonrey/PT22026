import type { Movie } from "@/types/api";

import { MovieCard } from "@/components/movies/MovieCard";

type MovieGridProps = { movies: Movie[]; priorityCount?: number };

export function MovieGrid({ movies, priorityCount = 0 }: MovieGridProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 md:grid-cols-4 xl:grid-cols-5">
      {movies.map((movie, index) => <MovieCard key={movie.id} movie={movie} priority={index < priorityCount} />)}
    </div>
  );
}
