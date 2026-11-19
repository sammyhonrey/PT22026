import { Search } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { MovieCollection } from "@/features/discovery/components/MovieCollection";
import { Pagination } from "@/features/discovery/components/Pagination";
import { useMovies } from "@/features/discovery/hooks";

const pageSize = 20;

export function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const submittedQuery = searchParams.get("q")?.trim() ?? "";
  const [query, setQuery] = useState(submittedQuery);
  const offset = Math.max(0, Number(searchParams.get("offset")) || 0);
  const movies = useMovies({ limit: pageSize, offset, q: submittedQuery || undefined }, Boolean(submittedQuery));

  useEffect(() => setQuery(submittedQuery), [submittedQuery]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = new URLSearchParams();
    if (query.trim()) next.set("q", query.trim());
    setSearchParams(next);
  }

  if (!submittedQuery) {
    return <div className="mx-auto max-w-3xl space-y-7 pt-6"><header><p className="text-sm font-bold text-[var(--mm-accent)]">Movie catalogue</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">Search Movies</h1></header><SearchForm query={query} setQuery={setQuery} submit={submit} /><EmptyState title="Search for a movie to get started" description="Enter a film title above to search the MovieMind catalogue." /></div>;
  }

  return (
    <div className="space-y-7"><header><p className="text-sm font-bold text-[var(--mm-accent)]">Movie catalogue</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">Search Results</h1></header><SearchForm isLoading={movies.isFetching} query={query} setQuery={setQuery} submit={submit} />
      <section aria-live="polite" aria-busy={movies.isFetching}><div className="mb-5 flex flex-wrap items-baseline justify-between gap-2"><h2 className="text-xl font-bold">Results for “{submittedQuery}”</h2>{movies.data && !movies.isFetching && <p className="text-sm text-[var(--mm-text-muted)]">{movies.data.total} {movies.data.total === 1 ? "movie" : "movies"}</p>}</div>
      <MovieCollection emptyDescription="Try a different title or shorter search phrase." emptyTitle="No movies found" error={movies.error} isLoading={movies.isFetching} movies={movies.data?.items} onRetry={() => void movies.refetch()} />
      {movies.data && !movies.isFetching && <Pagination limit={movies.data.limit} offset={movies.data.offset} onPageChange={(nextOffset) => setSearchParams({ offset: String(nextOffset), q: submittedQuery })} total={movies.data.total} />}</section></div>
  );
}

function SearchForm({ isLoading = false, query, setQuery, submit }: { isLoading?: boolean; query: string; setQuery: (value: string) => void; submit: (event: FormEvent<HTMLFormElement>) => void }) {
  return <form className="flex flex-col gap-3 sm:flex-row" onSubmit={submit} role="search"><label className="sr-only" htmlFor="movie-search">Search movies</label><div className="relative flex-1"><Search aria-hidden="true" className="pointer-events-none absolute left-4 top-1/2 size-5 -translate-y-1/2 text-[var(--mm-text-subtle)]" /><Input id="movie-search" className="pl-12" onChange={(event) => setQuery(event.target.value)} placeholder="Search for a movie..." value={query} /></div><Button className="rounded-xl" isLoading={isLoading} type="submit">Search</Button></form>;
}
