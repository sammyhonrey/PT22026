import { MovieCardSkeleton } from "@/components/movies/MovieCardSkeleton";

type MovieGridSkeletonProps = { count?: number };

export function MovieGridSkeleton({ count = 10 }: MovieGridSkeletonProps) {
  return (
    <div aria-busy="true" aria-label="Loading movies" className="grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 md:grid-cols-4 xl:grid-cols-5" role="status">
      {Array.from({ length: count }, (_, index) => <MovieCardSkeleton key={index} />)}
    </div>
  );
}
