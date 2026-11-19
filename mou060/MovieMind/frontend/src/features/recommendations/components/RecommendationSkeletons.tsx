import { Skeleton } from "@/components/ui/Skeleton";

export function RecommendationCardSkeleton({ featured = false }: { featured?: boolean }) {
  return (
    <div className={`overflow-hidden rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] ${featured ? "md:grid md:grid-cols-[11rem_minmax(0,1fr)]" : ""}`}>
      <Skeleton className={`aspect-[2/3] rounded-none ${featured ? "h-full min-h-64" : ""}`} />
      <div className={`space-y-3 p-4 ${featured ? "sm:p-6" : ""}`}>
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-6 w-4/5" />
        <Skeleton className="h-4 w-24" />
        <div className="space-y-2 pt-1"><Skeleton className="h-3 w-full" /><Skeleton className="h-3 w-5/6" /></div>
      </div>
    </div>
  );
}

export function RecommendationPreviewSkeleton() {
  return (
    <aside aria-busy="true" aria-label="Finding movies you'll love" className="recommendation-panel rounded-3xl border p-5 shadow-[var(--mm-card-shadow)] sm:p-6" role="status">
      <p className="text-xs font-bold uppercase tracking-[0.15em] text-[var(--mm-accent)]">Finding movies you&apos;ll love…</p>
      <h2 className="mt-3 text-xl font-black">Your Recommendations</h2>
      <div className="mt-5 space-y-3">
        {Array.from({ length: 3 }, (_, index) => (
          <div aria-hidden="true" className="flex gap-3 rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-subtle-surface)] p-2.5" key={index}>
            <Skeleton className="aspect-[2/3] w-16 shrink-0 rounded-xl" />
            <div className="min-w-0 flex-1 space-y-2 py-1"><Skeleton className="h-4 w-4/5" /><Skeleton className="h-3 w-1/4" /><Skeleton className="h-3 w-full" /></div>
          </div>
        ))}
      </div>
    </aside>
  );
}
