import { Search, Sparkles, X } from "lucide-react";
import { useQueries } from "@tanstack/react-query";
import { type FormEvent, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { fetchMovie } from "@/features/discovery/api";
import { useMovies, useGenres } from "@/features/discovery/hooks";
import { useMyPreferences, useSavePreferences, useGeneratePreferenceRecommendations } from "@/features/preferences/hooks";
import { RecommendationCard } from "@/features/recommendations/components/RecommendationCard";
import { RecommendationCardSkeleton } from "@/features/recommendations/components/RecommendationSkeletons";
import { isApiClientError } from "@/lib/api/errors";
import { queryKeys } from "@/lib/query/keys";
import type { Movie, Recommendations } from "@/types/api";

const defaultTraits = [
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Family",
  "Fantasy",
  "History",
  "Horror",
  "Mystery",
  "Romance",
  "Science Fiction",
  "Thriller",
  "War",
  "Western",
];

export function MoviePreferencesPage() {
  const preferences = useMyPreferences();
  const genres = useGenres();
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [favoriteMovieIds, setFavoriteMovieIds] = useState<string[]>([]);
  const [description, setDescription] = useState("");
  const [movieSearch, setMovieSearch] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null);

  const normalizedMovieSearch = movieSearch.trim();
  const movieSearchQuery = useMovies(
    { limit: 8, offset: 0, q: normalizedMovieSearch, sort: "title" },
    Boolean(normalizedMovieSearch),
  );
  const selectedMovieQueries = useQueries({
    queries: favoriteMovieIds.map((movieId) => ({
      queryKey: queryKeys.movies.detail(movieId),
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchMovie(movieId, signal),
      staleTime: 60_000,
    })),
  });

  const savePreferences = useSavePreferences();
  const generateRecommendations = useGeneratePreferenceRecommendations();

  const searchResults = movieSearchQuery.data?.items ?? [];

  const selectedMovies = useMemo(
    () => selectedMovieQueries.flatMap((query) => query.data ? [query.data] : []),
    [selectedMovieQueries],
  );
  const isLoadingSelectedMovies = selectedMovieQueries.some((query) => query.isFetching);
  const selectedMovieError = selectedMovieQueries.find((query) => query.error)?.error;

  const genreOptions = useMemo(() => {
    const names = new Set(
      genres.data?.map((genre) => genre.name === "Sci-Fi" ? "Science Fiction" : genre.name) ?? [],
    );
    defaultTraits.forEach((genre) => names.add(genre));
    return [...names].sort((left, right) => left.localeCompare(right));
  }, [genres.data]);

  useEffect(() => {
    const saved = preferences.data;
    if (!saved) return;
    setSelectedGenres(saved.favorite_genres ?? []);
    setFavoriteMovieIds(saved.favorite_movie_ids ?? []);
    setDescription(saved.description ?? "");
  }, [preferences.data]);

  function toggleGenre(genre: string) {
    setSelectedGenres((current) => current.includes(genre) ? current.filter((item) => item !== genre) : [...current, genre]);
  }

  function addFavoriteMovie(movie: Movie) {
    setFavoriteMovieIds((current) => (current.includes(movie.id) ? current : [...current, movie.id]));
    setMovieSearch("");
  }

  function removeFavoriteMovie(movieId: string) {
    setFavoriteMovieIds((current) => current.filter((id) => id !== movieId));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedDescription = description.trim();
    const payload = {
      favorite_genres: selectedGenres,
      favorite_movie_ids: favoriteMovieIds,
      description: normalizedDescription || null,
    };

    if (!payload.favorite_genres.length && !payload.favorite_movie_ids.length && !payload.description) {
      setError("Choose at least one genre, favorite movie, or brief description before generating recommendations.");
      return;
    }

    setError(null);
    try {
      await savePreferences.mutateAsync(payload);
      setRecommendations(null);
      const generated = await generateRecommendations.mutateAsync(payload);
      setRecommendations(generated);
    } catch (submitError) {
      const message = isApiClientError(submitError)
        ? submitError.message
        : submitError instanceof Error && submitError.message.trim()
          ? submitError.message
          : "We could not generate recommendations for your taste profile right now.";
      setError(message);
    }
  }

  const isLoading = savePreferences.isPending || generateRecommendations.isPending;

  if (!preferences.data && preferences.isFetching) {
    return (
      <div className="space-y-7">
        <PreferencesHeading />
        <div aria-busy="true" aria-label="Loading your movie preferences" className="space-y-5" role="status">
          <div className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6"><Skeleton className="h-6 w-40" /><div className="mt-4 flex flex-wrap gap-2">{Array.from({ length: 8 }, (_, index) => <Skeleton className="h-9 w-20 rounded-full" key={index} />)}</div></div>
          <div className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6"><Skeleton className="h-6 w-40" /><Skeleton className="mt-4 h-11 w-full rounded-2xl" /><Skeleton className="mt-4 h-8 w-36 rounded-full" /></div>
          <div className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-6"><Skeleton className="h-6 w-40" /><Skeleton className="mt-4 h-32 w-full rounded-2xl" /></div>
        </div>
      </div>
    );
  }

  if (!preferences.data && preferences.error) {
    return <div className="space-y-5"><PreferencesHeading /><StatusPanel tone="error">{isApiClientError(preferences.error) ? preferences.error.message : "We could not load your taste profile."}</StatusPanel><Button isLoading={preferences.isFetching} onClick={() => void preferences.refetch()} variant="secondary">Try again</Button></div>;
  }

  return (
    <div className="space-y-7">
      <PreferencesHeading />

      {preferences.error && <StatusPanel tone="error">We could not refresh your taste profile. Your saved preferences are still shown.</StatusPanel>}

      <form className="space-y-7" onSubmit={handleSubmit}>
        <section className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-5 shadow-[var(--mm-card-shadow)] sm:p-6">
          <h2 className="text-xl font-bold">Favorite genres</h2>
          <div className="mt-4 flex flex-wrap gap-2">
            {genreOptions.map((genre) => {
              const active = selectedGenres.includes(genre);
              return (
                <button
                  key={genre}
                  className={`rounded-full border px-3 py-2 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mm-accent)] ${active ? "border-transparent bg-[var(--mm-primary)] text-white shadow-[var(--mm-primary-shadow)] hover:bg-[var(--mm-primary-hover)]" : "border-[var(--mm-border)] bg-[var(--mm-surface-strong)] text-[var(--mm-text-muted)] hover:border-[var(--mm-primary)] hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)]"}`}
                  onClick={() => toggleGenre(genre)}
                  type="button"
                >
                  {genre}
                </button>
              );
            })}
          </div>
        </section>

        <section className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-5 shadow-[var(--mm-card-shadow)] sm:p-6">
          <h2 className="text-xl font-bold">Favorite movies</h2>
          <div className="mt-4 flex flex-col gap-3">
            <label className="sr-only" htmlFor="movie-preference-search">Search movies</label>
            <div className="relative">
              <Search aria-hidden="true" className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-[var(--mm-text-subtle)]" />
              <Input id="movie-preference-search" className="pl-12" onChange={(event) => setMovieSearch(event.target.value)} placeholder="Search the catalogue for movies you love" value={movieSearch} />
            </div>

            {normalizedMovieSearch && (
              <div aria-busy={movieSearchQuery.isFetching} className="rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] p-3">
                {movieSearchQuery.isFetching ? (
                  <div aria-label="Searching movies" className="space-y-3" role="status"><Skeleton className="h-10 w-full rounded-xl" /><Skeleton className="h-10 w-full rounded-xl" /></div>
                ) : movieSearchQuery.error ? (
                  <div className="space-y-3"><StatusPanel tone="error">{isApiClientError(movieSearchQuery.error) ? movieSearchQuery.error.message : "We could not search movies."}</StatusPanel><Button isLoading={movieSearchQuery.isFetching} onClick={() => void movieSearchQuery.refetch()} type="button" variant="secondary">Try again</Button></div>
                ) : searchResults.length ? (
                  <ul className="space-y-2">
                    {searchResults.map((movie) => (
                      <li key={movie.id} className="flex items-center justify-between gap-3 rounded-xl bg-[var(--mm-subtle-surface)] px-3 py-2">
                        <div>
                          <p className="font-semibold text-[var(--mm-text)]">{movie.title}</p>
                          <p className="text-xs text-[var(--mm-text-muted)]">{movie.genres.join(", ") || "No genres listed"}</p>
                        </div>
                        <Button className="rounded-xl px-3 py-2" disabled={favoriteMovieIds.includes(movie.id)} onClick={() => addFavoriteMovie(movie)} type="button" variant="secondary">
                          {favoriteMovieIds.includes(movie.id) ? "Added" : "Add"}
                        </Button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-[var(--mm-text-muted)]">No matching movies found. Try a different title or keyword.</p>
                )}
              </div>
            )}

            <div className="mt-2 flex flex-wrap gap-2">
              {selectedMovies.map((movie) => (
                <span key={movie.id} className="inline-flex items-center gap-2 rounded-full border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 py-2 text-sm text-[var(--mm-text)]">
                  {movie.title}
                  <button aria-label={`Remove ${movie.title}`} className="rounded-full p-1 text-[var(--mm-text-muted)] transition hover:bg-[var(--mm-hover-surface)] hover:text-[var(--mm-text)]" onClick={() => removeFavoriteMovie(movie.id)} type="button">
                    <X aria-hidden="true" className="size-3.5" />
                  </button>
                </span>
              ))}
              {isLoadingSelectedMovies && <div aria-label="Loading selected movies" className="flex gap-2" role="status">{favoriteMovieIds.map((id) => <Skeleton className="h-9 w-28 rounded-full" key={id} />)}</div>}
              {selectedMovieError && <div className="space-y-3"><StatusPanel tone="error">{isApiClientError(selectedMovieError) ? selectedMovieError.message : "We could not load one or more favorite movies."}</StatusPanel><Button isLoading={isLoadingSelectedMovies} onClick={() => selectedMovieQueries.filter((query) => query.error).forEach((query) => void query.refetch())} type="button" variant="secondary">Retry favorite movies</Button></div>}
              {!selectedMovies.length && !isLoadingSelectedMovies && !selectedMovieError ? (
                <p className="text-sm text-[var(--mm-text-muted)]">No favorite movies selected yet.</p>
              ) : null}
            </div>
          </div>
        </section>

        <section className="rounded-[28px] border border-[var(--mm-border)] bg-[var(--mm-surface)] p-5 shadow-[var(--mm-card-shadow)] sm:p-6">
          <h2 className="text-xl font-bold">Describe your taste</h2>
          <textarea
            className="mt-4 min-h-32 w-full rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-4 py-3 text-[var(--mm-text)] outline-none transition placeholder:text-[var(--mm-text-subtle)] focus:border-[var(--mm-accent)] focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)]/70"
            onChange={(event) => setDescription(event.target.value)}
            placeholder="I enjoy science fiction, psychological thrillers, intelligent stories, artificial intelligence, and unexpected plot twists."
            value={description}
          />
        </section>

        {genres.error && <div className="space-y-3"><StatusPanel tone="error">We could not refresh the genre list. You can still use the available preferences.</StatusPanel><Button isLoading={genres.isFetching} onClick={() => void genres.refetch()} type="button" variant="secondary">Retry genres</Button></div>}
        {error && <StatusPanel tone="error">{error}</StatusPanel>}

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm text-[var(--mm-text-muted)]">
            {selectedGenres.length || favoriteMovieIds.length || description.trim() ? "Your taste profile is ready to generate." : "Add a few preferences to start."}
          </div>
          <Button className="rounded-xl" disabled={isLoading} isLoading={isLoading} type="submit">
            Discover Movies For Me
          </Button>
        </div>
      </form>

      {generateRecommendations.isPending && (
        <section aria-busy="true" aria-label="Finding movies you'll love" className="space-y-5" role="status">
          <header><p className="text-sm font-bold text-[var(--mm-accent)]">Finding movies you&apos;ll love…</p><h2 className="mt-1 text-2xl font-bold">Your personalized results</h2></header>
          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 3 }, (_, index) => <RecommendationCardSkeleton key={index} />)}</div>
        </section>
      )}

      {recommendations && (
        <section className="space-y-5">
          <header className="flex items-center justify-between gap-3">
            <div>
              <p className="text-sm font-bold text-[var(--mm-accent)]">Your personalized results</p>
              <h2 className="mt-1 text-2xl font-bold">Movie matches for you</h2>
            </div>
            <span className="rounded-full border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--mm-text-muted)]">
              {recommendations.is_fallback ? "Popular fallback" : "Preference-based"}
            </span>
          </header>

          {recommendations.items.length ? (
            <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
              {recommendations.items.map((recommendation) => (
                <RecommendationCard key={recommendation.movie.id} recommendation={recommendation} />
              ))}
            </div>
          ) : (
            <StatusPanel tone="error">No recommendations matched your profile yet. Try broadening the genres, adding a few favorite movies, or describing more of what you enjoy.</StatusPanel>
          )}
        </section>
      )}
    </div>
  );
}

function PreferencesHeading() {
  return (
    <header>
      <p className="flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)]"><Sparkles aria-hidden="true" className="size-4" /> Movie preferences</p>
      <h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">My Movie Taste</h1>
      <p className="mt-2 max-w-2xl text-[var(--mm-text-muted)]">Tell MovieMind what you enjoy, and it will use the live content-based model to match a fresh recommendation set to your taste.</p>
    </header>
  );
}
