import { Search } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { MovieCollection } from "@/features/discovery/components/MovieCollection";
import { MovieFilters } from "@/features/discovery/components/MovieFilters";
import { Pagination } from "@/features/discovery/components/Pagination";
import { useGenres, useMovies } from "@/features/discovery/hooks";
import type { MovieListQuery } from "@/lib/query/keys";

const pageSize = 20;

function parseQuery(params: URLSearchParams): MovieListQuery {
  const year = Number(params.get("year"));
  const direction = params.get("direction");
  const sort = params.get("sort");
  return {
    direction: direction === "desc" ? "desc" : "asc",
    genre: params.get("genre") || undefined,
    limit: pageSize,
    offset: Math.max(0, Number(params.get("offset")) || 0),
    q: params.get("q") || undefined,
    sort: sort === "average_rating" || sort === "created_at" || sort === "ratings_count" || sort === "release_year" ? sort : "title",
    year: Number.isInteger(year) && year >= 1888 && year <= 2100 ? year : undefined,
  };
}

function updateParams(current: URLSearchParams, updates: Partial<MovieListQuery>) {
  const next = new URLSearchParams(current);
  for (const [key, value] of Object.entries(updates)) {
    if (value === undefined || value === "" || value === 0 && key === "offset") next.delete(key);
    else next.set(key, String(value));
  }
  return next;
}

export function DiscoverPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = parseQuery(searchParams);
  const [searchValue, setSearchValue] = useState(query.q ?? "");
  const movies = useMovies(query);
  const genres = useGenres();

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSearchParams(updateParams(searchParams, { q: searchValue.trim() || undefined, offset: 0 }));
  }

  function updateFilters(update: Partial<MovieListQuery>) {
    setSearchParams(updateParams(searchParams, { ...update, offset: 0 }));
  }

  function resetFilters() {
    setSearchParams(updateParams(searchParams, { direction: undefined, genre: undefined, offset: 0, sort: undefined, year: undefined }));
  }

  return (
    <div className="space-y-7">
      <header><p className="text-sm font-bold text-[var(--mm-accent)]">Movie catalogue</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">Discover Movies</h1><p className="mt-2 text-[var(--mm-text-muted)]">Find your next favorite film from the MovieMind catalogue.</p></header>
      <form className="flex flex-col gap-3 sm:flex-row" onSubmit={submitSearch} role="search">
        <label className="sr-only" htmlFor="discover-search">Search movies</label>
        <div className="relative flex-1"><Search aria-hidden="true" className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-[var(--mm-text-subtle)]" /><Input id="discover-search" className="pl-12" onChange={(event) => setSearchValue(event.target.value)} placeholder="Search titles..." value={searchValue} /></div>
        <Button className="rounded-xl" isLoading={movies.isFetching} type="submit">Search movies</Button>
      </form>
      <MovieFilters genres={genres.data ?? []} isLoadingGenres={genres.isLoading} onChange={updateFilters} onReset={resetFilters} value={query} />
      {genres.error && <div className="space-y-3"><StatusPanel tone="error">We could not load movie genres. You can still browse all movies.</StatusPanel><Button isLoading={genres.isFetching} onClick={() => void genres.refetch()} variant="secondary">Retry genres</Button></div>}
      <section aria-busy={movies.isFetching} aria-live="polite">
        <div className="mb-5 flex flex-wrap items-baseline justify-between gap-2"><h2 className="text-xl font-bold">{query.q ? `Results for “${query.q}”` : "Browse movies"}</h2>{movies.data && !movies.isFetching && <p className="text-sm text-[var(--mm-text-muted)]">{movies.data.total} {movies.data.total === 1 ? "movie" : "movies"}</p>}</div>
        <MovieCollection emptyDescription="Try another title, or clear filters to explore the full catalogue." emptyTitle="No movies match your search" error={movies.error} isLoading={movies.isFetching} movies={movies.data?.items} onRetry={() => void movies.refetch()} />
        {movies.data && !movies.isFetching && <Pagination limit={movies.data.limit} offset={movies.data.offset} onPageChange={(offset) => setSearchParams(updateParams(searchParams, { offset }))} total={movies.data.total} />}
      </section>
    </div>
  );
}
