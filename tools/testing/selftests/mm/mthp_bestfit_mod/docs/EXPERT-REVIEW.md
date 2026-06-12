# mthp_bestfit v4.0 — adversarial review against v6.12 mm
(the review I would expect from linux-mm; fix-forward items land in v4.1)

## Verdict up front

frag-fit solves the *fault-driven compaction* problem structurally — but
only under two conditions v4.0 does not itself guarantee (F1, F2 below).
The MemAvailable mechanism (boost-chain suppression) is real but one
link is less direct than presented (F9). Deferred splits are bounded by
the size menu, not solved (acknowledged). Decision cost is O(orders ×
slack) per fault and the refresh design is a power bug on a phone.
v4.1 makes the decision O(1), removes all idle wakeups, and closes the
two HIGH findings.

---

## F1 · HIGH · the 0..best mask contradicts the stall claim

`alloc_anon_folio()` in v6.12 computes `gfp = vma_thp_gfp_mask(vma)`
**once** and then walks the returned order bitmap top-down, attempting
each order. Our mask keeps bits 0..best inclusive — so every retained
bit is a full allocation attempt, and with `defrag = always` or
`madvise`+MADV_HUGEPAGE, each attempt carries __GFP_DIRECT_RECLAIM:
one decision can produce *several* compaction attempts. The shape that
was sold as "graceful staleness fallback" multiplies the exact event we
claim to prevent whenever the operator forgets `defrag=never`.

Fix (v4.1): `strict_mask=1` default — emit `BIT(best)` only. Miss ⇒
core falls to the implicit order-0 path in `do_anonymous_page` (one
large attempt, then base pages). Staleness now costs a 4 KB fault, not
a retry cascade. Legacy shape retained behind `strict_mask=0`.

## F2 · HIGH · permanent 100 ms kworker is a power regression

10 wakeups/s forever, screen-on idle included, to refresh a cache
nobody reads when no faults are occurring. Any Android power review
flags this on sight.

Fix (v4.1): delete the delayed work. Refresh is **self-clocked from the
fault path**: jiffies-gated, one CPU elected by `atomic_cmpxchg`, does
the ~40-load zone walk inline (sub-µs, lockless READ_ONCEs — safe under
per-VMA lock). Idle system ⇒ zero work. Fault storm ⇒ refresh rate
automatically tracks fault rate, which also shrinks the staleness
window exactly when it matters (see F3). The elegant part: the bug fix
and the herd mitigation are the same patch.

## F3 · MED · herd effect defeats exact_min within the window

10k faults/s against a 100 ms-stale shape: thousands of decisions all
see `free[4] = 9 ≥ 8` and all pick order 4; nine succeed. exact_min is
statistically meaningless under storm. F2's self-clocking shrinks the
window; v4.1 additionally keeps per-order **consumption budgets**
(`atomic_dec_if_positive` on decision, reset at refresh,
`herd_guard=1`): once the window's budget for an order is spent,
its eligibility bit is skipped locally. One atomic RMW per fault;
on ≤8-core phone silicon the cacheline bounce is cheaper than the
fallback storm it prevents.

## F4 · MED · nr_free is zone/migratetype/watermark/pcp-blind

`free_area[k].nr_free` sums all migratetypes: MIGRATE_CMA and
HIGHATOMIC reserves inflate the count with blocks a movable anon fault
may not get past `__zone_watermark_ok()`. Opposite direction at low
orders: pages parked in per-CPU lists (orders ≤ PAGE_ALLOC_COSTLY_ORDER)
are *not* in nr_free, so free[2..3] systematically undercounts.
Over-count is absorbed by the mask fallback; under-count costs hit
rate. 6.12 keeps no per-MT per-order counter, so the honest options are
(a) document, (b) per-zone vectors with watermark headroom subtracted.
v4.1 documents and leaves (b) as a follow-up — the fault has no zone
context at hook time anyway.

## F5 · MED · exec_boost is dead code on the path it targets

The hook matters at fault time for `do_anonymous_page` — anonymous
VMAs by definition. `vm_file && VM_EXEC` VMAs fault through the file
path; JIT regions are anon+exec but the classifier requires vm_file.
So the boost branch effectively never fires where it was designed to.
Keep it only if `by_type[exec]` proves nonzero via khugepaged-path
invocations; otherwise it is a per-fault test purchasing nothing.

## F6 · LOW · lifetime_aware is "first folio per VMA"

`anon_vma` is instantiated on the first write fault, so the demotion
applies to exactly one allocation per VMA lifetime — and never to
forked children (anon_vma inherited). Honest label: first-fault
demotion. Real lifetime policy needs VMA age, which the module can't
see. Keep, but stop overclaiming.

## F7 · LOW · counters record intent, not outcome

After our mask, `thp_vma_suitable_orders()` still applies
address-alignment filtering, and the buddy can still fail the order.
`order_hist` is "what bestfit decided," not "what got mapped."
Cross-check against per-size `stats/anon_fault_alloc` in every claim.

## F8 · LOW · the hook has no caller context

`thp_vma_allowable_orders()` is also consulted by khugepaged and
MADV_COLLAPSE paths; the vendor hook signature carries no tva_flags, so
fault-tuned thresholds silently govern collapse eligibility too. The
PMD bypass accidentally shields PMD collapse; sub-PMD collapse (none in
6.12 anon khugepaged) would not be. Document the blast radius.

## F9 · INFO · boost-chain link 1→2 is indirect — defend it precisely

`boost_watermark()` fires on cross-migratetype pageblock *steals*, not
on splits per se. Splits deplete high-order stock *within* owned
pageblocks; depletion is what later forces the fallback steal that
boosts. The chain is real, one hop longer than the slide implies. Say
"splits drain the stock whose exhaustion causes the boosting steals"
and no reviewer can touch it.

## F10 · INFO · frag-fit cannot manufacture contiguity

When the shape is depleted, suppressing is correct but mTHP coverage
goes to zero until frees/kcompactd rebuild stock. Pair with
`vm.compaction_proactiveness` ≈ 5–10 as the idle-time regenerator:
frag-fit stops the waste, proactive compaction slowly restocks the
shelves. Also: the module steers allocation only — munmap-side shape
evolution is untouched, by design.

---

## Complexity: v4.0 → v4.1

| component | v4.0 | v4.1 |
|---|---|---|
| ceiling | O(P) loop, mul per iter | O(1): `fls_long(vma_size / min_pages) − 1 − PAGE_SHIFT` |
| pass 1 + pass 2 | O(P) + O(P·(s+1)) ≈ 32 loads | **O(1): one 64-bit load + two fls** — eligibility precomputed as two packed 16-bit bitmaps (exact, bounded) in a single word; selection = `fls(bm & ceiling_mask)`; single-word publish makes the snapshot atomically consistent for free |
| stats | preempt_disable/enable + 12 branches | `this_cpu_inc/add`, no preempt toggling |
| refresh | periodic kworker, 10 wakes/s idle | fault-clocked election; 0 idle cost; O(zones·orders) ≈ 40 loads amortized once per window, paid by one fault |
| hot state | 10 × u64 cache + work item | 8 B packed word (hot) + u32[10] raw (debugfs) + 10 atomics (budgets) |
| herd exposure | full window | budgets bound consumption per window |

The deeper point: the buddy's segregated free lists already performed
the best-fit *sort*. Selection over an 8-bucket histogram should never
have been a scan — precompute eligibility at write time (rare), make
the read path (hot) a mask-and-fls. That is the canonical kernel shape
for this problem.

## What I would still not sign off without

Three-boot data (baseline / frag_fit=0 / frag_fit=1) on the 1.5 GB
target with the F7 cross-check, a screen-on-idle power trace proving
F2's fix (zero bestfit wakeups in `kworker` accounting), and the
order-histogram time series showing the shape isn't being eaten
level-by-level into permanent suppression (F10 interaction).
