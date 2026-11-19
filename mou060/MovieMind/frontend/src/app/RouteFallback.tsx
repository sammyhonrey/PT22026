import { PageLoader } from "@/components/ui/PageLoader";

export function RouteFallback() {
  return <PageLoader label="Loading this part of MovieMind…" />;
}
