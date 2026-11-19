type SkeletonProps = { className?: string };

export function Skeleton({ className = "" }: SkeletonProps) {
  return <div aria-hidden="true" className={`animate-pulse rounded bg-[var(--mm-skeleton)] ${className}`} />;
}
