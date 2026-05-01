.. SPDX-License-Identifier: GPL-2.0

==================================================
mthp_bestfit — VMA-size Best-fit mTHP Order Policy
==================================================

:Author: Suyog
:Version: 3
:Config: ``CONFIG_MTHP_BESTFIT``


Overview
========

Multi-size Transparent Huge Pages (mTHP) allow the Linux kernel to allocate
anonymous and file-backed memory in hugepage orders between 2 (16 KB) and
``PMD_ORDER`` (2 MB on x86-64), rather than only at the classic 2 MB boundary.

The baseline kernel's mTHP strategy is **greedy highest-order-first**: when a
page fault occurs, the kernel tries the largest policy-allowed order first and
falls back to smaller orders only if allocation fails.  This approach has five
measurable deficiencies for production systems and for memory-constrained
devices:

1. **Wasted allocation attempts** — for a 256 KB VMA with ``always`` enabled
   for all orders, the kernel tries 2 MB, 1 MB, 512 KB, all of which fail the
   geometric fit check, before landing on the correct 256 KB order.  Every
   failed attempt increments ``MTHP_STAT_ANON_FAULT_FALLBACK`` and wastes CPU.

2. **No VMA-type differentiation** — ELF ``.text`` segments and anonymous
   malloc heaps receive identical order treatment, despite the fact that iTLB
   coverage is far more critical for executable code than for data.

3. **Uncapped stack VMAs** — fully-initialised stack VMAs receive PMD_ORDER
   (2 MB) hugepage attempts even though stacks grow unpredictably, leading to
   expensive hugepage splits later.

4. **No memory pressure feedback** — on 1–2 GB devices the buddy allocator may
   have no contiguous pages at large orders.  The kernel attempts the order
   anyway, triggering costly memory compaction.

5. **No NUMA locality preference** — on multi-node systems, selecting an order
   available only on a remote NUMA node incurs higher latency than choosing a
   smaller order available locally.

``mthp_bestfit`` corrects all five deficiencies through a **pluggable function
pointer hook** (``mthp_order_filter_fn``) that the core kernel calls at the
end of ``__thp_vma_allowable_orders()``.  No existing kernel code paths are
altered beyond the single hook invocation.


Architecture
============

Hook Integration
----------------

``mthp_bestfit`` registers itself via a single exported function pointer in
``mm/huge_memory.c``::

    /* mm/huge_memory.c */
    unsigned long (*mthp_order_filter_fn)(...) __read_mostly;
    EXPORT_SYMBOL(mthp_order_filter_fn);

The hook is called at the **end** of ``__thp_vma_allowable_orders()`` for all
non-zero returns where the call context is neither ``TVA_SMAPS`` nor
``TVA_FORCED_COLLAPSE``::

    fn = READ_ONCE(mthp_order_filter_fn);
    if (fn)
        orders = fn(vma, vm_flags, type, orders);

This means:

- ``TVA_SMAPS``: **not intercepted** — ``/proc/PID/smaps`` always shows the
  kernel's true policy eligibility, not a filtered view.
- ``TVA_FORCED_COLLAPSE``: **not intercepted** — ``MADV_COLLAPSE`` represents
  explicit user intent and must not be overridden.
- ``TVA_PAGEFAULT``: **intercepted** — the primary hot path.
- ``TVA_KHUGEPAGED``: **intercepted** — khugepaged collapse decisions.

Additionally, khugepaged's VMA scan loop (``mm/khugepaged.c``) consults the
filter **before** scanning a VMA for PMD-order collapse.  If the filter would
suppress PMD_ORDER for a given VMA, the scan is skipped entirely, saving the
full ``khugepaged_scan_pmd()`` cost.

Safe Module Load / Unload
--------------------------

Registration uses ``WRITE_ONCE`` and unregistration uses ``WRITE_ONCE(NULL)``
followed by ``synchronize_rcu()``, ensuring no CPU can execute a stale pointer
after ``module_exit()`` returns.  The module can be loaded and unloaded at
runtime without rebooting::

    insmod mthp_bestfit.ko   # activates immediately
    rmmod mthp_bestfit       # restores baseline behaviour immediately


Policies Implemented
====================

1. VMA-size Geometric Best-fit
-------------------------------

The core algorithm selects the **largest order** where at least ``min_pages``
complete, hugepage-aligned regions fit within the VMA.  Alignment is computed
from the first naturally-aligned address ``>= vm_start``, so the result is
correct for any fault address within the VMA::

    first_aligned = ALIGN(vm_start, page_size);
    fits = (vm_end - first_aligned) >= min_pages * page_size;

This is strictly more accurate than a raw ``vma_size >= N * page_size`` check,
which ignores alignment waste at the start of the VMA.

Once the geometric ceiling is determined, the returned bitmask masks out all
orders strictly above it::

    new_orders = allowed_orders & ((1UL << (best_order + 1)) - 1);

For large VMAs (``>= PMD_ORDER``), the bitmask is returned unchanged, leaving
the allocator free to choose at the top.

2. Stack VMA Cap (order 2 = 16 KB)
------------------------------------

Stack VMAs (``VM_GROWSDOWN`` set, ``VM_STACK_INCOMPLETE_SETUP`` cleared) are
hard-capped at order 2 (16 KB).  The kernel's ``__thp_vma_allowable_orders()``
allows fully-initialised stacks to receive PMD-order hugepages, which fragment
badly when the stack grows into adjacent pages and the hugepage must be split.

A 16 KB hugepage provides TLB benefit on hot stack frames while reserving
large contiguous blocks for VMAs that use them more uniformly.

3. Exec Segment Boost (+1 order)
----------------------------------

File-backed executable VMAs (ELF ``.text`` segments, ``VM_EXEC | vm_file``)
receive a +1 order promotion after the geometric fit passes.  The motivation:

- The iTLB (Instruction TLB) covers fewer entries than the dTLB on all modern
  microarchitectures (e.g., Intel Ice Lake: 128 iTLB entries vs 2048 L2 dTLB).
- Each iTLB entry covers one hugepage.  A larger hugepage doubles coverage per
  entry, directly reducing iTLB miss rates on hot code paths.
- The kernel applies identical order treatment to code and data VMAs despite
  this asymmetry.

The boost is bounded: ``order < PMD_ORDER`` prevents overflow above the
maximum supported order.  It can be disabled independently via sysctl.

4. Memory Pressure Awareness
------------------------------

After the geometric best-fit order is computed, the module checks the buddy
allocator's ``free_area[]`` to determine whether any populated zone has free
blocks at that order::

    for_each_populated_zone(zone)
        if (zone->free_area[order].nr_free > 0)
            return true;  /* order is available */

If the order is unavailable in all zones, the module scans downward for the
highest order that **is** available and uses that instead.

**Why this matters for low-memory devices:**

On a 1 GB or 2 GB system running an embedded Linux workload:

- PMD_ORDER (2 MB) requires 512 physically contiguous 4 KB pages.  After
  the system has been running for minutes, memory fragmentation makes this
  extremely unlikely without compaction.
- Requesting an unavailable order triggers ``kcompactd`` or synchronous
  compaction in the page allocator, adding 10–100 ms latency spikes to
  application page faults.
- ``mthp_bestfit`` avoids this by selecting the highest order that the buddy
  allocator can already satisfy without compaction.

The ``free_area[]`` read is intentionally racy (no zone lock held): the goal
is a cheap heuristic snapshot, not a guarantee.  A stale read causes at most
a single sub-optimal selection; correctness is maintained by the allocator's
own fallback path.

5. NUMA Locality Awareness
---------------------------

On multi-node NUMA systems, the module checks whether the **current CPU's
local NUMA node** has free pages at the desired order before committing::

    pgdat = NODE_DATA(numa_node_id());
    for each zone in pgdat:
        if zone->free_area[order].nr_free > 0: return true

If the local node cannot satisfy the order, the module downgrades to the
largest order available locally.  This preference for local allocation:

- Avoids cross-node memory access latency (typically 50–150 ns vs 10–30 ns
  locally).
- Reduces NUMA migration pressure by placing hugepages where the faulting CPU
  is most likely to access them.
- On UMA systems the check always returns ``true`` (nid=0, single pgdat),
  adding negligible overhead.

6. VMA Lifetime Awareness
--------------------------

Brand-new VMAs (``vma->anon_vma == NULL``) have never been faulted.  Their
final size is unknown at fault time.  For these VMAs, the module reduces the
selected order by 1 before returning::

    if (lifetime_aware && !vma->anon_vma && order > 2)
        order--;

**Why this matters:**

- On a 1 GB device, a 1 MB hugepage reserved for a VMA that only ever uses
  16 KB is 63 pages of wasted contiguous physical memory.
- Established VMAs (``anon_vma != NULL``) have already demonstrated their
  access pattern and deserve full best-fit order selection.
- The reduction is one order only, preserving mTHP benefit while conserving
  contiguous memory for VMAs that need it.

7. khugepaged Scan Guidance
----------------------------

``khugepaged`` — the kernel daemon that proactively collapses PTEs into huge
pages — currently only scans VMAs at ``PMD_ORDER``.  The module's filter is
consulted **before** the scan begins::

    fn = READ_ONCE(mthp_order_filter_fn);
    if (fn && !fn(vma, vm_flags, TVA_KHUGEPAGED, BIT(PMD_ORDER))) {
        cc->progress++;
        continue;  /* skip: filter suppresses PMD_ORDER for this VMA */
    }

This prevents khugepaged from spending ``khugepaged_scan_pmd()`` time on VMAs
the filter would suppress anyway, saving CPU cycles in the background daemon.

8. Dry-run Mode
----------------

Setting ``mthp_bestfit_dry_run=1`` enables full decision tracking and
statistics collection **without** modifying the return value of
``__thp_vma_allowable_orders()``.  The kernel behaves exactly as if the module
were not loaded, while the module accumulates statistics.

Use this on production systems to measure the expected impact before enabling
full interception.


Sysctl Reference
================

All knobs are under ``/proc/sys/vm/``.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Default
     - Description
   * - ``mthp_bestfit_enabled``
     - 1
     - Master on/off.  0 disables all filtering; kernel baseline resumes.
   * - ``mthp_bestfit_min_pages``
     - 2
     - Minimum number of complete aligned hugepages that must fit within
       the VMA before an order is selected.  Range: 1–16.
   * - ``mthp_bestfit_exec_boost``
     - 1
     - Enable +1 order promotion for file-backed exec VMAs (ELF ``.text``).
   * - ``mthp_bestfit_dry_run``
     - 0
     - Count decisions without modifying return values.  Safe profiling mode.
   * - ``mthp_bestfit_pressure_aware``
     - 1
     - Downgrade order when buddy allocator has no free blocks at that order.
       Prevents compaction on low-memory devices.
   * - ``mthp_bestfit_numa_aware``
     - 1
     - Prefer orders available on the local NUMA node.  No-op on UMA systems.
   * - ``mthp_bestfit_lifetime_aware``
     - 1
     - Reduce order by 1 for brand-new VMAs (never faulted).  Conserves large
       contiguous blocks on memory-constrained devices.


DebugFS Reference
=================

Statistics are available at::

    /sys/kernel/debug/mthp_bestfit/stats

Example output::

    mthp_bestfit v3 — VMA-size best-fit mTHP order selection

    Configuration
      enabled:        1
      min_pages:      2
      exec_boost:     1
      dry_run:        0
      pressure_aware: 1
      numa_aware:     1
      lifetime_aware: 1

    Decision summary
      total:               1428571
      suppressed:          83214   (5%)
      pressure_downgraded: 42100   (2%)
      numa_downgraded:     18300   (1%)
      lifetime_conserved:  61200   (4%)

    By VMA type
      stack  : 12041   (0%)
      exec   : 234104  (16%)
      anon   : 987321  (69%)
      file   : 195105  (13%)

    Order histogram
      order  2 (    16 KB): 112030  (7%)
      order  5 (   128 KB): 312400  (21%)
      order  6 (   256 KB): 441020  (30%)
      order  8 (  1024 KB): 289310  (20%)
      order  9 (  2048 KB): 273811  (19%)

``pressure_downgraded`` and ``numa_downgraded`` are per-decision counters
(a single decision can trigger both) and are additive with the order histogram.


Comparison with Baseline Kernel
================================

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Dimension
     - Baseline Linux kernel
     - mthp_bestfit
   * - Order selection strategy
     - Greedy highest-first; fallback on allocation failure
     - Precomputed geometric best-fit; impossible orders never attempted
   * - Wasted allocation attempts
     - N iterations for a VMA that only fits order N
     - 0 iterations for impossible orders
   * - VMA size awareness
     - None — flat global bitmask per policy
     - Per-fault alignment-correct computation
   * - Alignment correctness
     - Checked reactively inside allocator loop
     - Proactive: ``ALIGN(vm_start, page_size)`` before selection
   * - min-coverage enforcement
     - Does not exist
     - Configurable 1–16 pages; enforces N complete pages
   * - Stack VMA handling
     - Uncapped — PMD_ORDER attempted
     - Hard cap at order 2 (16 KB)
   * - Exec segment preference
     - None
     - +1 order boost for ``VM_EXEC | vm_file``
   * - Memory pressure awareness
     - None — compaction triggered on failure
     - Proactive buddy check; downgrade before compaction
   * - NUMA locality preference
     - None
     - Local-node availability check before order selection
   * - VMA lifetime signals
     - None
     - Conservative on new VMAs (anon_vma == NULL)
   * - khugepaged guidance
     - Scans all policy-allowed VMAs at PMD_ORDER
     - Skips VMAs the filter would suppress; saves scan CPU
   * - TVA_SMAPS protection
     - N/A (no interception)
     - Explicitly skipped — smaps shows true policy
   * - TVA_FORCED_COLLAPSE respect
     - N/A
     - Explicitly skipped — MADV_COLLAPSE honoured
   * - Runtime enable/disable
     - Sysfs policy change or recompile
     - ``echo 0 > /proc/sys/vm/mthp_bestfit_enabled``
   * - Safe profiling mode
     - None
     - ``dry_run=1`` — statistics without modification
   * - Decision statistics
     - Allocation outcomes only (fault_alloc, fallback)
     - Decisions by order, VMA type, pressure, NUMA, lifetime
   * - Kernel modification required
     - — (N/A)
     - Zero existing code paths altered; one hook call added
   * - Runtime load/unload
     - — (N/A)
     - Full ``insmod``/``rmmod`` with safe pointer teardown
   * - Architecture portability
     - — (N/A)
     - Dynamic ``(PAGE_SIZE << i)``; correct on all arches


Low-Memory Device Impact (1–2 GB DDR)
======================================

Embedded and mobile targets with 1–2 GB of physical RAM — such as development
boards running Linux with DDR3/DDR4 at this capacity — see the largest benefit
from ``mthp_bestfit`` for several compounding reasons.

Memory Fragmentation is Chronic
---------------------------------

On a 1 GB system running a full userspace (init, daemons, browser, multimedia),
the physical address space fragments within minutes.  PMD-order (2 MB) blocks
become unavailable without compaction long before the system is memory-
pressured.  The baseline kernel triggers ``kcompactd`` or synchronous
compaction on every PMD-order page fault that cannot be satisfied directly.

``mthp_bestfit`` with ``pressure_aware=1`` avoids this entirely: it selects
the highest order the buddy allocator can already satisfy and never requests an
order that would trigger compaction.

Compaction Cost Is Proportionally Higher
-----------------------------------------

Compaction on a 1 GB system scans proportionally more of total RAM than on a
32 GB server, because a 2 MB compaction target represents 0.2% of 1 GB but
only 0.006% of 32 GB.  The latency spikes from compaction (10–100 ms) are
significant for interactive and real-time workloads.

By selecting achievable orders without compaction, ``mthp_bestfit`` eliminates
this latency class entirely for workloads where it is most damaging.

Contiguous Memory Is a Shared Resource
----------------------------------------

On small systems, large contiguous blocks are needed by:

- DMA-capable drivers (camera, GPU, audio codec)
- CMA allocations for multimedia pipelines
- GPU command buffers
- Kernel hugepages for critical daemons

Wasting a 2 MB contiguous block on a VMA that only needed 128 KB directly
reduces the contiguous memory available to these consumers.  The
``lifetime_aware`` policy conserves large contiguous blocks for VMAs that are
verified to need them, by giving brand-new VMAs a conservative initial order.

mTHP Enables Sub-PMD Benefit on 1–2 GB Systems
-------------------------------------------------

The baseline kernel's 2 MB PMD hugepage is rarely achievable on fragmented
1 GB systems.  Before mTHP, this meant most anonymous memory fell back to
4 KB base pages with no TLB benefit.  With ``mthp_bestfit``, orders 2–6
(16 KB–256 KB) are almost always achievable even on a fragmented 1 GB system,
providing meaningful TLB coverage without requiring large contiguous blocks.

The result is a spectrum of TLB improvement previously unavailable on small
systems:

.. list-table::
   :widths: 15 15 20 50
   :header-rows: 1

   * - Order
     - Size
     - Pages needed
     - Achievability on 1 GB system
   * - 2
     - 16 KB
     - 4
     - Almost always available; fragmentation-immune
   * - 3
     - 32 KB
     - 8
     - Highly available
   * - 4
     - 64 KB
     - 16
     - Generally available
   * - 5
     - 128 KB
     - 32
     - Available after moderate runtime
   * - 6
     - 256 KB
     - 64
     - Available early in system lifetime
   * - 7
     - 512 KB
     - 128
     - Available on fresh boot or post-drop-caches
   * - 9
     - 2 MB
     - 512
     - Requires compaction; often unavailable

Organizational Benefits
========================

For organizations shipping products on Linux with memory-constrained targets:

**Reduced compaction-induced latency spikes**
  Interactive applications (browser, media player, UI) see fewer and shorter
  stalls from background compaction.  p99 page-fault latency improves on
  fragmented systems.

**Higher effective hugepage coverage**
  By selecting achievable sub-PMD orders, a far greater fraction of anonymous
  memory benefits from TLB-miss reduction than the PMD-only baseline achieves
  on small systems.

**No kernel rebuild required**
  The module is a ``CONFIG_MTHP_BESTFIT=m`` loadable module.  It can be
  deployed, tuned, and withdrawn on a running system without rebooting —
  essential for fleet management.

**Profiling before commitment**
  ``dry_run=1`` allows measuring the decision distribution on real production
  workloads before enabling modification.  This de-risks deployment completely.

**Transparent observability**
  ``/sys/kernel/debug/mthp_bestfit/stats`` provides per-order, per-VMA-type,
  and per-adjustment-reason breakdowns that the kernel's own ``mthp_stats``
  does not expose.

**Works alongside existing kernel mTHP policy**
  The ``always``/``madvise``/``inherit``/``never`` per-order sysfs policy
  remains fully effective.  ``mthp_bestfit`` applies **after** the kernel's
  policy filter, so it can only make the selection more conservative, never
  more permissive.


Configuration Examples
======================

Conservative (1 GB embedded device, interactive workload)::

    echo 1 > /proc/sys/vm/mthp_bestfit_enabled
    echo 2 > /proc/sys/vm/mthp_bestfit_min_pages
    echo 1 > /proc/sys/vm/mthp_bestfit_pressure_aware
    echo 1 > /proc/sys/vm/mthp_bestfit_lifetime_aware
    echo 0 > /proc/sys/vm/mthp_bestfit_exec_boost

Balanced (2 GB device, general-purpose server)::

    echo 1 > /proc/sys/vm/mthp_bestfit_enabled
    echo 2 > /proc/sys/vm/mthp_bestfit_min_pages
    echo 1 > /proc/sys/vm/mthp_bestfit_exec_boost
    echo 1 > /proc/sys/vm/mthp_bestfit_pressure_aware
    echo 1 > /proc/sys/vm/mthp_bestfit_numa_aware
    echo 1 > /proc/sys/vm/mthp_bestfit_lifetime_aware

Profiling only (any system, no impact)::

    echo 1 > /proc/sys/vm/mthp_bestfit_enabled
    echo 1 > /proc/sys/vm/mthp_bestfit_dry_run
    # Read /sys/kernel/debug/mthp_bestfit/stats after workload run
    # Then disable dry_run once decision distribution looks correct


Known Limitations and Future Work
===================================

**khugepaged sub-PMD collapse**
  khugepaged currently collapses only to ``PMD_ORDER``.  A full sub-PMD
  anonymous collapse path would allow khugepaged to proactively promote
  medium-sized VMAs to orders 2–8, currently only reachable on page fault.
  This requires a new ``collapse_anon_subpmd()`` function in
  ``mm/khugepaged.c`` that works at the PTE level rather than the PMD level.

**Adaptive pressure threshold**
  The current pressure check is binary (available / unavailable).  A
  per-order free-page count threshold (e.g., "only select order N if at
  least K free blocks exist") would allow finer-grained backoff before
  fragmentation becomes critical.

**Memory cgroup awareness**
  The pressure check is system-wide.  For memcg-constrained containers,
  the relevant pressure is the cgroup's ``memory.usage_in_bytes`` vs
  ``memory.limit_in_bytes``, not global free pages.

**VMA age timestamps**
  VMA lifetime is currently approximated by ``anon_vma`` presence.  A
  real age (``vma->vm_start_time``) would allow order promotion after a
  VMA has been stable for a configurable duration.

**Writeback and reclaim interaction**
  Large hugepages that are partially dirty are expensive to reclaim.  A
  writeback-rate-aware order cap would reduce reclaim pressure under
  write-heavy workloads.
