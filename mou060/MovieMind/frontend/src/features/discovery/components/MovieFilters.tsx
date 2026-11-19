import { SlidersHorizontal, X } from "lucide-react";

import { Button } from "@/components/ui/Button";
import type { Genre } from "@/types/api";
import type { MovieListQuery } from "@/lib/query/keys";

type MovieFiltersProps = {
  genres: Genre[];
  isLoadingGenres: boolean;
  onChange: (next: Partial<MovieListQuery>) => void;
  onReset: () => void;
  value: MovieListQuery;
};

export function MovieFilters({ genres, isLoadingGenres, onChange, onReset, value }: MovieFiltersProps) {
  const hasFilters = Boolean(value.genre || value.year || (value.sort && value.sort !== "title") || (value.direction && value.direction !== "asc"));

  return (
    <section aria-label="Movie filters" className="flex flex-wrap items-end gap-3 rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] p-4">
      <div className="flex items-center gap-2 pb-2 text-sm font-semibold text-[var(--mm-text-muted)]"><SlidersHorizontal aria-hidden="true" className="size-4 text-[var(--mm-accent)]" /> Filters</div>
      <label aria-busy={isLoadingGenres} className="grid min-w-36 flex-1 gap-1 text-xs font-semibold text-[var(--mm-text-muted)]">Genre
        {isLoadingGenres && <span className="text-[var(--mm-text-subtle)]">Loading genres…</span>}
        <select className="min-h-10 rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 text-sm text-[var(--mm-text)] outline-none transition-[border-color,box-shadow,opacity] duration-200 focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)] disabled:cursor-not-allowed disabled:opacity-60" disabled={isLoadingGenres} onChange={(event) => onChange({ genre: event.target.value || undefined })} value={value.genre ?? ""}>
          <option value="">All genres</option>
          {genres.map((genre) => <option key={genre.id} value={genre.name}>{genre.name}</option>)}
        </select>
      </label>
      <label className="grid min-w-28 flex-1 gap-1 text-xs font-semibold text-[var(--mm-text-muted)]">Year
        <input className="min-h-10 rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 text-sm text-[var(--mm-text)] outline-none focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)]" max="2100" min="1888" onChange={(event) => onChange({ year: event.target.value ? Number(event.target.value) : undefined })} placeholder="Any year" type="number" value={value.year ?? ""} />
      </label>
      <label className="grid min-w-34 flex-1 gap-1 text-xs font-semibold text-[var(--mm-text-muted)]">Sort by
        <select className="min-h-10 rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 text-sm text-[var(--mm-text)] outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)]" onChange={(event) => onChange({ sort: event.target.value as NonNullable<MovieListQuery["sort"]> })} value={value.sort ?? "title"}>
          <option value="title">Title</option><option value="average_rating">Average rating</option><option value="ratings_count">Most rated</option><option value="release_year">Release year</option>
        </select>
      </label>
      <label className="grid min-w-28 flex-1 gap-1 text-xs font-semibold text-[var(--mm-text-muted)]">Order
        <select className="min-h-10 rounded-xl border border-[var(--mm-border)] bg-[var(--mm-surface-strong)] px-3 text-sm text-[var(--mm-text)] outline-none transition-[border-color,box-shadow] duration-200 focus-visible:ring-2 focus-visible:ring-[var(--mm-accent)]" onChange={(event) => onChange({ direction: event.target.value as "asc" | "desc" })} value={value.direction ?? "asc"}>
          <option value="asc">Ascending</option><option value="desc">Descending</option>
        </select>
      </label>
      {hasFilters && <Button className="min-h-10 rounded-xl px-3" onClick={onReset} variant="ghost"><X aria-hidden="true" className="size-4" /> Clear</Button>}
    </section>
  );
}
