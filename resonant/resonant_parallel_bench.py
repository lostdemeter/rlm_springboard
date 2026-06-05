#!/usr/bin/env python3
"""
Parallel Benchmark: Resonant Array vs Traditional Deduplication.

Shows that the Resonant Array is embarrassingly parallel across:
  1. Files (independent phase computation per file)
  2. Heads (independent measurements per Riemann zero)
  3. Queries (independent resonance checks per candidate)

Compare:
  - Traditional: compare every file pair O(N²) with shared state
  - Resonant Array: compute phases O(N·H) in parallel, no shared state

Run: python3 resonant_parallel_bench.py
"""

import os
import sys
import math
import time
import hashlib
from multiprocessing import Pool, cpu_count

TWOPI = 2.0 * math.pi
GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073]

def content_hash(path):
    try:
        with open(path, 'rb') as f:
            return int(hashlib.sha256(f.read()).hexdigest()[:16], 16)
    except:
        return 0

def compute_phases(args):
    """Compute phases for one file across all heads. Fully independent."""
    path, gammas = args
    key = content_hash(path)
    if key == 0:
        return None
    return [((gamma * key) % TWOPI) for gamma in gammas]

def compare_pair(args):
    """Compare two files' full content for deduplication. O(content_length)."""
    path_a, path_b = args
    try:
        with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
            if fa.read() == fb.read():
                return (path_a, path_b)
    except:
        pass
    return None

def heading(s):
    print(f"\n{'='*70}")
    print(f"  {s}")
    print(f"{'='*70}")


def benchmark():
    repo = "/home/thorin/Documents/OpenCode/phi_geist"
    
    print("█" * 70)
    print("  PARALLEL BENCHMARK: Resonant Array vs Traditional Dedup")
    print(f"  Embarrassingly parallel across files, heads, and queries")
    print("█" * 70)
    
    # Gather files
    files = []
    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'logs'}]
        for fn in filenames:
            files.append(os.path.join(root, fn))
    
    n = len(files)
    print(f"\n  Files: {n}")
    print(f"  CPUs: {cpu_count()}")
    
    # ── Approach 1: Traditional pairwise comparison ──
    heading("Traditional Deduplication: O(N²) Pairwise Content Comparison")
    
    # Time how long it takes to compare just 200 pairs (estimating full O(N²))
    if n >= 2:
        sample_pairs = [(files[i], files[j]) 
                       for i in range(min(20, n)) 
                       for j in range(i+1, min(20, n))]
        t_start = time.time()
        for pair in sample_pairs:
            compare_pair(pair)
        t_sample = time.time() - t_start
        
        total_pairs = n * (n - 1) // 2
        estimated_full = t_sample * total_pairs / len(sample_pairs)
        print(f"  Sampled {len(sample_pairs)} pairs: {t_sample:.4f}s")
        print(f"  Full O(N²) for {total_pairs:,} pairs: ~{estimated_full:.1f}s")
        print(f"  Requires shared file system access (all files must be readable)")
        print(f"  Requires O(N²) I/O — reading every pair into memory")
    
    # ── Approach 2: Resonant Array single-threaded ──
    heading("Resonant Array: Single-thread Phase Computation O(N·H)")
    
    t_start = time.time()
    phases = {}
    for path in files:
        key = content_hash(path)
        if key != 0:
            phases[path] = [((gamma * key) % TWOPI) for gamma in GAMMAS[:4]]
    t_single = time.time() - t_start
    
    print(f"  Computed phases for {len(phases)} files: {t_single:.4f}s")
    print(f"  Each file computes independently — no pairwise I/O needed")
    print(f"  Rate: {len(phases)/t_single:.0f} files/second")
    
    # ── Approach 3: Resonant Array multi-process parallel ──
    heading("Resonant Array: Multi-process Parallel O(N·H)")
    
    t_start = time.time()
    with Pool(processes=cpu_count()) as pool:
        gammas = GAMMAS[:4]
        parallel_phases = pool.map(compute_phases, [(f, gammas) for f in files])
    t_parallel = time.time() - t_start
    
    valid = [p for p in parallel_phases if p is not None]
    print(f"  Computed phases for {len(valid)} files: {t_parallel:.4f}s")
    print(f"  Rate: {len(valid)/t_parallel:.0f} files/second")
    print(f"  Speedup: {t_single/t_parallel:.1f}x over single-threaded")
    
    # ── Query Benchmark ──
    heading("Query Throughput: Independent Resonance Checks")
    
    if len(valid) >= 100:
        sample_key = content_hash(files[0])
        sample_phases = [((gamma * sample_key) % TWOPI) for gamma in GAMMAS[:4]]
        
        t_start = time.time()
        n_queries = 1000
        n_hits = 0
        for _ in range(n_queries):
            threshold = 0.15
            for p in parallel_phases[:100]:
                if p is not None:
                    d = min(abs(p[0] - sample_phases[0]), 
                           TWOPI - abs(p[0] - sample_phases[0])) / math.pi
                    if d < threshold:
                        n_hits += 1
        t_query = time.time() - t_start
        
        items_checked = n_queries * 100
        print(f"  Checked {items_checked:,} resonance comparisons in {t_query:.4f}s")
        print(f"  Rate: {items_checked/t_query:,.0f} checks/second")
        print(f"  Hits: {n_hits}")
        print(f"  Each check: {t_query/items_checked*1e6:.2f} µs")
    
    # ── Summary ──
    heading("Summary: Why This Matters")
    print(f"""
  OPERATION          TRADITIONAL              RESONANT ARRAY
  ─────────────────   ──────────────────────  ──────────────────────────
  Index N files       O(N²) pairwise I/O      O(N·H) phase compute
  Parallelize         Hard (shared state)     TRIVIAL (independent phases)
  Scale to 10⁶ files  ~5 × 10¹¹ comparisons   ~4 × 10⁶ phase computes
  Add a node          Must re-coordinate      Just compute same formula
  GPU acceleration    Hard (branching I/O)    One FMA per phase
  Seed coordination   Required                NOTHING — ζ zeros are universal
  Memory per node     O(N) file data          O(N·H) phases (8 bytes each)
""")
    
    # Show the parallelism math
    n_files = 1_000_000
    n_heads = 4
    ops = n_files * n_heads
    
    heading(f"Projected: 1 Million Files")
    print(f"  Total phase computations: {ops:,}")
    print(f"  Single CPU (10⁶ files/s): {ops/1e6:.1f}s")
    print(f"  32 CPUs: {ops/1e6/32:.1f}s")
    print(f"  GPU (1M threads, 1 FMA each): ~{ops/1e6:.1f}ms — effectively real-time")
    print(f"\n  Traditional O(N²) comparison for 1M files:")
    print(f"  Pairs: {n_files*(n_files-1)//2:,}")
    print(f"  Time at 1M pairs/sec: {n_files*(n_files-1)//2/1e6:.0f}s = "
          f"{n_files*(n_files-1)//2/1e6/3600:.0f} hours")
    print(f"\n  The Resonant Array isn't just faster — it's PARALLELIZABLE.")
    print(f"  Traditional dedup is fundamentally limited by pairwise I/O.")


if __name__ == "__main__":
    benchmark()
