import { apiClient } from "@/lib/api/client";
import type { MovieListQuery } from "@/lib/query/keys";
import type { Genre, Movie, MovieList } from "@/types/api";

export async function fetchMovies(query: MovieListQuery, signal?: AbortSignal) {
  const { data } = await apiClient.get<MovieList>("/movies", { params: query, signal });
  return data;
}

export async function fetchGenres(signal?: AbortSignal) {
  const { data } = await apiClient.get<Genre[]>("/movies/genres", { signal });
  return data;
}

export async function fetchMovie(movieId: string, signal?: AbortSignal) {
  const { data } = await apiClient.get<Movie>(`/movies/${movieId}`, { signal });
  return data;
}

export async function fetchPopularMovies(limit: number, signal?: AbortSignal) {
  const { data } = await apiClient.get<Movie[]>("/movies/popular", { params: { limit }, signal });
  return data;
}
