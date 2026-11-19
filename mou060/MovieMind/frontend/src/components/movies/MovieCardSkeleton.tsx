import { Skeleton } from "@/components/ui/Skeleton";

type MovieCardSkeletonProps = { className?: string };

export function MovieCardSkeleton({ className = "" }: MovieCardSkeletonProps) {
  return (
    <div className={`overflow-hidden rounded-2xl border border-[var(--mm-border)] bg-[var(--mm-surface)] ${className}`}>
      <Skeleton className="aspect-[2/3] rounded-none" />
      <div className="space-y-2 p-3.5">
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-3 w-2/5" />
      </div>
    </div>
  );
}
