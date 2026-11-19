import { Bookmark, BookmarkCheck } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { useAuth } from "@/features/auth/AuthProvider";
import { useAddToWatchlist, useRemoveFromWatchlist, useWatchlist } from "@/features/watchlist/hooks";
import { isApiClientError } from "@/lib/api/errors";

type WatchlistButtonProps = { compact?: boolean; movieId: string; movieTitle?: string };

export function WatchlistButton({ compact = false, movieId, movieTitle }: WatchlistButtonProps) {
  const { isAuthenticated } = useAuth();
  const list = useWatchlist(0, 100, isAuthenticated);
  const add = useAddToWatchlist();
  const remove = useRemoveFromWatchlist();
  const [message, setMessage] = useState<string | null>(null);
  const inWatchlist = list.data?.items.some((item) => item.movie.id === movieId) ?? false;
  const pending = add.isPending || remove.isPending;

  const actionLabel = inWatchlist
    ? `Remove ${movieTitle ?? "movie"} from My List`
    : `Add ${movieTitle ?? "movie"} to My List`;

  if (!isAuthenticated) {
    return (
      <Link aria-label={compact ? `Sign in to save ${movieTitle ?? "movie"}` : undefined} to="/login">
        <Button className={compact ? "size-11 rounded-xl p-0" : ""} variant="secondary">
          <Bookmark aria-hidden="true" className="size-4" />
          {!compact && "Sign in to save"}
        </Button>
      </Link>
    );
  }

  async function toggle() {
    setMessage(null);
    try {
      if (inWatchlist) await remove.mutateAsync(movieId);
      else await add.mutateAsync(movieId);
      setMessage(inWatchlist ? "Removed from your list." : "Added to your list.");
    } catch (error) {
      setMessage(isApiClientError(error) ? error.message : "We could not update your list.");
    }
  }

  return <div className="space-y-2"><Button aria-label={compact ? actionLabel : undefined} className={compact ? "size-11 rounded-xl p-0" : ""} disabled={pending || list.isFetching} isLoading={pending || list.isFetching} onClick={() => void toggle()} variant={inWatchlist ? "secondary" : "primary"}>{inWatchlist ? <BookmarkCheck aria-hidden="true" className="size-4" /> : <Bookmark aria-hidden="true" className="size-4" />}{!compact && (inWatchlist ? "Saved to My List" : "Add to My List")}</Button>{message && <p className={`mm-content-enter text-sm ${message.includes("could not") || message.includes("Too many") ? "text-[var(--mm-danger)]" : "text-[var(--mm-success)]"}`} role="status">{message}</p>}</div>;
}
