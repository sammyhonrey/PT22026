import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { fetchGenres, fetchMovie, fetchMovies, fetchPopularMovies } from "@/features/discovery/api";
import { queryKeys, type MovieListQuery } from "@/lib/query/keys";

const movieStaleTime = 60_000;

export function useMovies(query: MovieListQuery, enabled = true) {
  return useQuery({
    queryKey: queryKeys.movies.list(query),
    queryFn: ({ signal }) => fetchMovies(query, signal),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: movieStaleTime,
  });
}

export function useGenres() {
  return useQuery({
    queryKey: queryKeys.movies.genres(),
    queryFn: ({ signal }) => fetchGenres(signal),
    staleTime: 5 * 60_000,
  });
}

export function useMovie(movieId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.movies.detail(movieId),
    queryFn: ({ signal }) => fetchMovie(movieId, signal),
    enabled,
    staleTime: movieStaleTime,
  });
}

export function usePopularMovies(limit = 5) {
  return useQuery({
    queryKey: queryKeys.movies.popular(limit),
    queryFn: ({ signal }) => fetchPopularMovies(limit, signal),
    staleTime: movieStaleTime,
  });
}
