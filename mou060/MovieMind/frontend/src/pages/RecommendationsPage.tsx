import { RefreshCw, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { StatusPanel } from "@/components/ui/StatusPanel";
import { RecommendationCard } from "@/features/recommendations/components/RecommendationCard";
import { RecommendationCardSkeleton } from "@/features/recommendations/components/RecommendationSkeletons";
import { useMyRecommendations } from "@/features/recommendations/hooks";
import { isApiClientError } from "@/lib/api/errors";

export function RecommendationsPage() {
  const recommendations = useMyRecommendations();
  if (recommendations.isFetching) return <div aria-busy="true" className="space-y-7"><PageHeading /><section><h2 className="mb-4 text-xl font-bold">Top pick</h2><RecommendationCardSkeleton featured /></section><section><h2 className="mb-4 text-xl font-bold">More for you</h2><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 3 }, (_, index) => <RecommendationCardSkeleton key={index} />)}</div></section><p className="sr-only" role="status">Finding movies you&apos;ll love…</p></div>;
  if (recommendations.error) {
    const message = isApiClientError(recommendations.error) ? recommendations.error.message : "We could not load your recommendations.";
    return <div className="space-y-5"><PageHeading /><StatusPanel tone="error">{message}</StatusPanel><Button onClick={() => void recommendations.refetch()} variant="secondary"><RefreshCw aria-hidden="true" className="size-4" /> Try again</Button></div>;
  }
  const data = recommendations.data;
  if (!data?.items.length) return <div className="space-y-5"><PageHeading /><EmptyState title="No recommendations available yet" description="Rate a few movies to help MovieMind learn what you enjoy, then return here for fresh suggestions." action={<Link to="/discover"><Button>Discover Movies</Button></Link>} /></div>;
  const [featured, ...rest] = data.items;
  const heading = data.is_fallback ? "Popular picks while you get started" : "Recommended for You";
  const subtitle = data.is_fallback ? "These are popularity-based suggestions until you have rated at least five movies." : "These suggestions are based on your ratings and interests.";

  return <div className="space-y-7"><header><p className="flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)]"><Sparkles aria-hidden="true" className="size-4" /> {data.source}</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">{heading}</h1><p className="mt-2 max-w-2xl text-[var(--mm-text-muted)]">{subtitle}</p></header><section aria-labelledby="featured-recommendation"><h2 className="mb-4 text-xl font-bold" id="featured-recommendation">Top pick</h2><RecommendationCard featured recommendation={featured} /></section>{rest.length > 0 && <section aria-labelledby="more-recommendations"><h2 className="mb-4 text-xl font-bold" id="more-recommendations">More for you</h2><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{rest.map((recommendation) => <RecommendationCard key={recommendation.movie.id} recommendation={recommendation} />)}</div></section>}</div>;
}

function PageHeading() {
  return <header><p className="flex items-center gap-2 text-sm font-bold text-[var(--mm-accent)]"><Sparkles aria-hidden="true" className="size-4" /> Your movie matches</p><h1 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">Recommended for You</h1></header>;
}
