#!/usr/bin/env python3
"""
Resonant Array Oracle — interactive demo.

Insert text snippets, query by resonance, see the phase space.
Demonstrates: deterministic lookup, seedless uniformity,
multi-head resonance, and frequency signatures.

Controls:
  i <text>    — insert a snippet
  q <text>    — query by resonance
  s <key>     — show frequency signature
  h           — toggle head view (cycle)
  t <0-1>     — set resonance threshold
  d           — dump all stored items
  ?           — help
  quit/exit   — leave
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TWOPI = 2.0 * math.pi
GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073]

# ── Phase primitive ───────────────────────────────────────────────────

def phase(key: int, gamma: float) -> float:
    return (gamma * key) % TWOPI

def circ_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def hash_key(text: str) -> int:
    """Deterministic key from text content."""
    h = 0
    for c in text:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h

# ── Resonant Array (simplified, embed-friendly) ───────────────────────

class ResonantArray:
    def __init__(self, n_heads=4, n_buckets=256):
        self.H = min(n_heads, len(GAMMAS))
        self.B = n_buckets
        self.gammas = GAMMAS[:self.H]
        self.buckets = [[[] for _ in range(self.B)] for _ in range(self.H)]
        self.items = {}  # key → (text, metadata)

    def insert(self, key: int, text: str, meta=""):
        self.items[key] = (text, meta)
        for h in range(self.H):
            p = phase(key, self.gammas[h])
            b = int(p / TWOPI * self.B) % self.B
            # Collision chain
            for i, (k, v) in enumerate(self.buckets[h][b]):
                if k == key:
                    self.buckets[h][b][i] = (key, (text, meta))
                    break
            else:
                self.buckets[h][b].append((key, (text, meta)))

    def lookup(self, key: int):
        return self.items.get(key)

    def resonate(self, key: int, threshold: float = 0.15):
        """Find all items with phase within threshold in ANY head."""
        key_phases = [phase(key, g) for g in self.gammas]
        results = []
        for ok, (otext, ometa) in self.items.items():
            if ok == key:
                continue
            best_dist = 1.0
            best_h = -1
            for h in range(self.H):
                d = circ_dist(key_phases[h], phase(ok, self.gammas[h]))
                if d < best_dist:
                    best_dist = d
                    best_h = h
            if best_dist < threshold:
                results.append((ok, otext, ometa, best_h, best_dist))
        results.sort(key=lambda x: x[4])
        return results

    def signature(self, key: int):
        """H-element complex signature."""
        return [complex(math.cos(phase(key, g)), math.sin(phase(key, g)))
                for g in self.gammas[:self.H]]

    def stats(self):
        occs = [sum(len(b) for b in hb) for hb in self.buckets]
        return {'items': len(self.items), 'heads': self.H,
                'buckets': self.B, 'occupancy': occs}

# ── ASCII Phase Circle ────────────────────────────────────────────────

def draw_circle(phases, highlight_phase=None, highlight_range=None, title="Phase Space"):
    """ASCII art unit circle with marked phases."""
    w, h = 36, 18
    cx, cy = w // 2, h // 2
    r = min(cx, cy) - 2
    grid = [[' ' for _ in range(w)] for _ in range(h)]
    
    # Draw circle
    for angle in range(0, 360, 5):
        rad = angle * math.pi / 180
        x = int(cx + r * math.cos(rad))
        y = int(cy + r * math.sin(rad))
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = '.'
    
    # Mark highlight range
    if highlight_phase is not None and highlight_range is not None:
        for angle in range(0, 360):
            rad = angle * math.pi / 180
            p = rad
            d = circ_dist(p, highlight_phase)
            if d < highlight_range:
                x = int(cx + r * math.cos(rad))
                y = int(cy + r * math.sin(rad))
                if 0 <= x < w and 0 <= y < h:
                    grid[y][x] = '·'
    
    # Mark items
    for p in phases:
        x = int(cx + r * math.cos(p))
        y = int(cy + r * math.sin(p))
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = '●'
    
    # Mark highlight
    if highlight_phase is not None:
        x = int(cx + r * math.cos(highlight_phase))
        y = int(cy + r * math.sin(highlight_phase))
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = '★'
    
    # Center label
    lines = [''.join(row) for row in grid]
    label = f"  {title}  "
    label_x = cx - len(label) // 2
    for i, c in enumerate(label):
        if 0 <= label_x + i < w:
            grid[h//2][label_x + i] = c
    
    return '\n'.join(''.join(row) for row in grid)

# ── Main ──────────────────────────────────────────────────────────────

def main():
    ra = ResonantArray(n_heads=4, n_buckets=256)
    current_head = 0
    threshold = 0.15
    
    def show_help():
        print("""
  Commands:
    i <text>    — insert snippet (keyed by hash of text)
    q <text>    — query by resonance (finds similar items)
    s <key>     — show frequency signature for key
    l <key>     — lookup exact key
    h           — cycle through heads (0-3)
    t <0-1>     — set resonance threshold
    d           — dump all items with phases
    stats       — show array statistics
    ?           — this help
    quit        — exit
""")
    
    # Seed with starter items
    starters = [
        "the golden ratio is everywhere in nature",
        "prime numbers are the atoms of arithmetic",
        "riemann zeros follow random matrix statistics",
        "attention is all you need",
        "transformers are universal function approximators",
        "phi squared equals phi plus one",
        "the montgomery odlyzko law connects primes to physics",
        "neural networks learn distributed representations",
        "the music box principle generalizes phi ladder systems",
        "signed integer attention replaces softmax",
    ]
    for text in starters:
        k = hash_key(text)
        ra.insert(k, text, "starter")
    
    print("█" * 60)
    print("  RESONANT ARRAY ORACLE")
    print("  γ·key mod 2π → deterministic resonance")
    print(f"  {len(starters)} starter items, {ra.H} heads, {ra.B} buckets")
    print("  Type ? for help")
    print("█" * 60)
    
    while True:
        try:
            cmd = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if not cmd:
            continue
        
        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if action in ('quit', 'exit'):
            break
        
        elif action == '?':
            show_help()
        
        elif action == 'i':
            key = hash_key(arg)
            ra.insert(key, arg)
            sig = ra.signature(key)
            print(f"  Inserted at key={key}")
            print(f"  Head 0 phase: {phase(key, ra.gammas[0]):.4f} rad")
            print(f"  Signature |c₀| = {abs(sig[0]):.4f}")
        
        elif action == 'q':
            if not arg:
                print("  Usage: q <text>")
                continue
            qkey = hash_key(arg)
            qphases = [phase(qkey, g) for g in ra.gammas]
            
            # Show the query's phases
            print(f"\n  Query key={qkey} phases:")
            for h in range(ra.H):
                print(f"    Head {h} (γ={ra.gammas[h]:.4f}): {qphases[h]:.4f} rad")
            
            # Show phase circle for current head
            all_phases = []
            for k in ra.items:
                all_phases.append(phase(k, ra.gammas[current_head]))
            print(f"\n{draw_circle(all_phases, qphases[current_head], threshold, 
                  f'Head {current_head} — ★ = query, ● = stored, · = resonance band')}")
            
            # Find resonances
            results = ra.resonate(qkey, threshold)
            
            if not results:
                print(f"  No resonances within threshold={threshold:.2f}")
                print(f"  Try increasing threshold with t <value>")
            else:
                print(f"  {len(results)} resonances (threshold={threshold:.2f}):")
                print(f"  {'Key':>12s} {'Head':>4s} {'Dist':>6s} {'Text':>40s}")
                print(f"  {'-'*12} {'-'*4} {'-'*6} {'-'*40}")
                for k, t, m, h, d in results[:8]:
                    display = t[:38] + '..' if len(t) > 40 else t
                    print(f"  {k:>12d} {h:>4d} {d:>6.3f} {display}")
                if len(results) > 8:
                    print(f"  ... and {len(results)-8} more")
        
        elif action == 's':
            try:
                skey = int(arg)
            except ValueError:
                skey = hash_key(arg)
            sig = ra.signature(skey)
            print(f"  Signature for key={skey}:")
            for h, c in enumerate(sig):
                deg = math.degrees(math.atan2(c.imag, c.real)) % 360
                print(f"    Head {h}: {c.real:+.4f} + {c.imag:+.4f}i  |={abs(c):.4f}  ∠{deg:.1f}°")
        
        elif action == 'l':
            try:
                lkey = int(arg)
            except ValueError:
                lkey = hash_key(arg)
            item = ra.lookup(lkey)
            if item:
                print(f"  key={lkey}: \"{item[0]}\" [{item[1]}]")
                for h in range(ra.H):
                    p = phase(lkey, ra.gammas[h])
                    print(f"    Head {h}: phase={p:.4f} rad, bucket={int(p/TWOPI*ra.B)%ra.B}")
            else:
                print(f"  key={lkey} not found")
        
        elif action == 'h':
            current_head = (current_head + 1) % ra.H
            print(f"  Viewing head {current_head} (γ={ra.gammas[current_head]:.4f})")
        
        elif action == 't':
            try:
                threshold = max(0.0, min(1.0, float(arg)))
                print(f"  Threshold set to {threshold:.2f}")
            except ValueError:
                print(f"  Usage: t <0-1>")
        
        elif action == 'd':
            print(f"  All {len(ra.items)} items:")
            for k, (t, m) in sorted(ra.items.items()):
                phases_str = " ".join(f"{phase(k, ra.gammas[h]):.2f}" for h in range(ra.H))
                print(f"    key={k:>10d}  phases=[{phases_str}]  \"{t[:50]}\"")
        
        elif action == 'stats':
            s = ra.stats()
            print(f"  Items: {s['items']}")
            print(f"  Heads: {s['heads']}")
            print(f"  Buckets per head: {s['buckets']}")
            for h in range(s['heads']):
                occupied = sum(1 for b in ra.buckets[h] if b)
                max_chain = max(len(b) for b in ra.buckets[h])
                mean = s['occupancy'][h] / s['buckets']
                print(f"    Head {h}: mean={mean:.2f}, max_chain={max_chain}, "
                      f"occupied_buckets={occupied}/{s['buckets']}")
        
        else:
            print(f"  Unknown: {action}. Type ? for help.")


if __name__ == "__main__":
    main()
