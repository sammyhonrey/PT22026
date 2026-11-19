import { Bookmark } from "lucide-react";
import { Link } from "react-router-dom";

import { MovieGrid } from "@/components/movies/MovieGrid";
import { MovieGridSkeleton } from "@/components/movies/MovieGridSkeleton";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { Pagination } from "@/features/discovery/components/Pagination";
import { useWatchlist } from "@/features/watchlist/hooks";
import { isApiClientError } from "@/lib/api/errors";
import { useState } from "react";

const pageSize = 20;

export function WatchlistPage() {
  const [offset, setOffset] = useState(0);
  const list = useWatchlist(offset, pageSize);
  return <div className="space-y-7"><header><p className="flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)]"><Bookmark aria-hidden="true" className="size-4" /> Personal collection</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">My List</h1><p className="mt-2 text-[var(--mm-text-muted)]">Movies you have saved to revisit later.</p></header>{list.isLoading ? <MovieGridSkeleton /> : list.error ? <div className="space-y-4"><StatusPanel tone="error">{isApiClientError(list.error) ? list.error.message : "We could not load your list."}</StatusPanel><Button onClick={() => void list.refetch()} variant="secondary">Try again</Button></div> : !list.data?.items.length ? <EmptyState title="Your list is empty" description="Save movies from a detail page to collect them here." action={<Link to="/discover"><Button>Discover Movies</Button></Link>} /> : <><div className="flex justify-end text-sm text-[var(--mm-text-muted)]">{list.data.total} saved {list.data.total === 1 ? "movie" : "movies"}</div><MovieGrid movies={list.data.items.map((item) => item.movie)} priorityCount={5} /><Pagination limit={list.data.limit} offset={list.data.offset} onPageChange={setOffset} total={list.data.total} /></>}</div>;
}
