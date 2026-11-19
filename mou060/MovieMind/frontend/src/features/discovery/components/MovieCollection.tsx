import { RefreshCw } from "lucide-react";

import { MovieGrid } from "@/components/movies/MovieGrid";
import { MovieGridSkeleton } from "@/components/movies/MovieGridSkeleton";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { isApiClientError } from "@/lib/api/errors";
import type { Movie } from "@/types/api";

type MovieCollectionProps = { error: unknown; isLoading: boolean; movies: Movie[] | undefined; onRetry: () => void; emptyDescription: string; emptyTitle: string };

export function MovieCollection({ emptyDescription, emptyTitle, error, isLoading, movies, onRetry }: MovieCollectionProps) {
  if (isLoading) return <MovieGridSkeleton />;
  if (error) {
    const message = isApiClientError(error) ? error.message : "We could not load movies. Please try again.";
    return <div className="space-y-4"><StatusPanel tone="error">{message}</StatusPanel><Button onClick={onRetry} variant="secondary"><RefreshCw aria-hidden="true" className="size-4" /> Try again</Button></div>;
  }
  if (!movies?.length) return <EmptyState title={emptyTitle} description={emptyDescription} />;
  return <MovieGrid movies={movies} priorityCount={5} />;
}
