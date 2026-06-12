# mthp_bestfit v4.0 "frag-fit" — design notes

## The inversion

v3.3: VMA size picks the order, buddy state can only veto/demote.
v4.0: the buddy free-list **shape** (per-order nr_free vector) picks the
order; VMA size is only the ceiling so sparse VMAs still can't
over-allocate.

Per fault, within [order 2, ceiling] ∩ allowed_orders:

1. **Exact fit** — highest order with ≥ exact_min_blocks free at exactly
   that order. The fault consumes a hole of its own size: zero splits,
   fragmentation is eaten, every larger block survives.
2. **Bounded fit** — highest order satisfiable from blocks within
   +split_slack orders (hard-capped below PMD_ORDER). Splitting a k+1/k+2
   block is cheap and never touches costly contiguity.
3. **Suppress** — the shape fits nothing → base pages. Our faults are
   structurally incapable of triggering compaction.

The free-shape vector is a delayed-work cache (refresh_ms=100, one
nr_free read per order per zone — negligible). Staleness is safe twice
over: the mask returned is 0..best inclusive, so core mm's
alloc_anon_folio walks down the remaining orders itself, and the buddy
fallback catches anything else. Side effect worth knowing: v3.x's
buddy_guard read the monitor's INT_MAX "unavailable" sentinel on GKI
and never fired; v4.0 sources per-order data locally, so the legacy
path's guard is now real too.

## Why fewer splits ⇒ more free memory (the actual mechanism)

Allocation placement doesn't change MemFree arithmetic directly — the
chain is indirect and it's the one that bites 1.5 GB targets:

order-k fault with empty order-k list → buddy splits k+1..k+n →
high-order lists drain → next high-order/pageblock demand falls back
across migratetypes → **external fragmentation event** →
`watermark_boost_factor` boosts zone watermarks → kswapd wakes and
reclaims **ahead of need** → page cache and idle anon get evicted →
MemAvailable drops and pgmajfault/pgsteal rise.

Exact-fit steering removes the first link. No split traffic → no
fallback steals → no boost → kswapd quiet → reclaim stops eating your
working set. That, plus min_pages bounding RSS over-allocation, is the
honest path from "fill the holes" to "more free available memory."

## Knob semantics

- `frag_fit` 1/0 — v4 shape-driven vs full v3.3 legacy pipeline. One
  sysctl flips the A/B; counters distinguish the modes.
- `exact_min_blocks` (8) — hole population required before an order is
  picked. Raise to be choosier (more base-page fallback), lower to
  harvest scarcer holes.
- `split_slack` (2) — how far above k pass-2 may split from. 0 = pure
  hole-filling, never split anything.
- `refresh_ms` (100) — shape cache staleness bound.
- `min_pages`, `exec_boost` (ceiling +1), `lifetime_aware` (ceiling −1),
  stack cap, PMD Step-1 bypass — all retained, all now act on the
  ceiling, not the result.

## Verification chain (tie each claim to a counter)

- Steering works: debugfs `stats` → Exact-fit % high, order histogram
  tracking the "Buddy free shape" table rather than VMA sizes.
- Compaction: /proc/vmstat compact_stall, compact_fail ≈ flat during
  fault storms; `Frag suppressed` counts the stalls that didn't happen.
- Free memory: MemAvailable steady under pressure; pgsteal_kswapd and
  pgscan deltas down vs baseline (the watermark-boost chain at rest).
  Cross-check /sys/kernel/debug/extfrag/extfrag_index drifting toward
  -1/low values at orders 2–5 and /proc/pagetypeinfo movable blocks
  stable.
- Splits: per-size stats/split_deferred and thp_deferred_split_page
  deltas down (frag-fit doesn't directly fix partial-unmap splits, but
  smaller, shape-matched folios shrink the exposure).

Run ab-compare.sh on three boots if you want the full story:
baseline / frag_fit=0 / frag_fit=1 — that isolates the inversion's
contribution from the v3.3 gating's.

## Limits, stated plainly

Exact-fit per order, not per zone/migratetype: the cache sums zones, so
a "hole" may be in a zone or migratetype the fault can't use — the
0..best mask shape absorbs the miss. Pass-1 prefers the largest fitting
order (perf-leaning); a strict smallest-hole-first policy would consume
fragmentation faster at TLB cost — flip the pass-1 scan direction if
you want to experiment. And frag-fit governs allocation-side
fragmentation only; Scudo-driven partial-unmap splits remain bounded by
the size menu, not eliminated.
