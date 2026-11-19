import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/Button";

type PaginationProps = { limit: number; offset: number; onPageChange: (offset: number) => void; total: number };

export function Pagination({ limit, offset, onPageChange, total }: PaginationProps) {
  if (total <= limit) return null;
  const page = Math.floor(offset / limit) + 1;
  const pageCount = Math.ceil(total / limit);
  const lastOffset = Math.max(0, (pageCount - 1) * limit);

  return (
    <nav aria-label="Movie result pages" className="mt-8 flex items-center justify-center gap-4">
      <Button aria-label="Previous page" className="min-h-10 rounded-xl px-3" disabled={offset === 0} onClick={() => onPageChange(Math.max(0, offset - limit))} variant="secondary"><ChevronLeft aria-hidden="true" className="size-4" /> Previous</Button>
      <p className="text-sm text-[var(--mm-text-muted)]">Page <span className="font-bold text-[var(--mm-text)]">{page}</span> of {pageCount}</p>
      <Button aria-label="Next page" className="min-h-10 rounded-xl px-3" disabled={offset + limit >= total} onClick={() => onPageChange(Math.min(lastOffset, offset + limit))} variant="secondary">Next <ChevronRight aria-hidden="true" className="size-4" /></Button>
    </nav>
  );
}
