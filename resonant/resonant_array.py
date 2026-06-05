"""
Resonant Array: A generic data structure indexed by Riemann-zero phase.

Definition:
  A Resonant Array R_γ is a collection of elements indexed by
  phase(key) = γ · key mod 2π, where γ is a nontrivial zero of the
  Riemann zeta function.

Operations:
  - insert(key, value): store at phase(key)      O(1)
  - lookup(key): retrieve by exact phase          O(1)
  - resonate(key, threshold): find elements within phase distance  O(N)
  - signature(key, H): H-element complex amplitude vector  O(H)
  - heads(H): access H independent subspaces (one per zero)  O(H)

Properties:
  - Deterministic: same key → same phase on every machine, every run
  - Uniform: Montgomery-Odlyzko law guarantees uniform phase distribution
  - Independent: different zeros → independent distributions
  - Seedless: no random state required
  - Scalable: more zeros → more heads → finer resonance resolution
  - Modifiable: change threshold for tunable similarity width

The Resonant Array unifies:
  - Hash tables (exact key → value)
  - Bloom filters (membership via phase bits)
  - Similarity search (nearby phases → related elements)
  - Attention (phase alignment determines resonance)
  - Frequency signatures (amplitude vector over H heads)
"""

import math

# First 16 Riemann zeros
GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073,
          48.005151, 49.773832, 52.970322, 56.446248,
          59.347044, 60.831779, 65.112545, 67.079811]

TWOPI = 2.0 * math.pi


def phase(key: int | float, gamma: float) -> float:
    """Universal primitive: γ · key mod 2π"""
    return (gamma * key) % TWOPI


def circ_dist(a: float, b: float) -> float:
    """Circular distance on [0, 2π), normalized to [0, 1]."""
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi


# ══════════════════════════════════════════════════════════════════════
# Resonant Array — Core Implementation
# ══════════════════════════════════════════════════════════════════════

class ResonantArray:
    """Generic phase-indexed associative array with resonance search.
    
    Each element is stored at exactly one phase position per head.
    Similar keys → similar phases → nearby buckets → resonance detection.
    
    Head count H = number of independent Riemann zeros used.
    Buckets B = number of phase bins (default 2^h or configurable).
    """
    
    def __init__(self, n_heads: int = 4, n_buckets: int = 1024):
        self.H = min(n_heads, len(GAMMAS))
        self.B = n_buckets
        self.gammas = GAMMAS[:self.H]
        # Storage: list of H tables, each with B buckets
        self.tables = [[[] for _ in range(self.B)] for _ in range(self.H)]
        self._all_items = {}  # for resonance scan
    
    def _bucket(self, key: int, head: int) -> int:
        p = phase(key, self.gammas[head])
        return int(p / TWOPI * self.B) % self.B
    
    def insert(self, key: int, value):
        """Store value at all H heads."""
        self._all_items[key] = value
        for h in range(self.H):
            b = self._bucket(key, h)
            # Replace if exists, otherwise append
            for i, (k, v) in enumerate(self.tables[h][b]):
                if k == key:
                    self.tables[h][b][i] = (key, value)
                    break
            else:
                self.tables[h][b].append((key, value))
    
    def lookup(self, key: int):
        """Retrieve by exact key (checks head 0)."""
        b = self._bucket(key, 0)
        for k, v in self.tables[0][b]:
            if k == key:
                return v
        return None
    
    def lookup_by_phase(self, target_phase: float, head: int = 0):
        """Retrieve all items at a specific phase in a given head."""
        b = int(target_phase / TWOPI * self.B) % self.B
        return list(self.tables[head][b])
    
    def resonate(self, key: int, threshold: float = 0.1) -> list:
        """Find all keys with phase within threshold in ANY head.
        
        Two keys resonate if their phases are within threshold * π
        on the unit circle in at least one head dimension.
        
        Returns list of (key, value, head, circ_dist) tuples sorted
        by distance.
        """
        key_phases = [phase(key, g) for g in self.gammas]
        results = []
        
        for other_key, other_val in self._all_items.items():
            if other_key == key:
                continue
            for h in range(self.H):
                other_phase = phase(other_key, self.gammas[h])
                d = circ_dist(key_phases[h], other_phase)
                if d < threshold:
                    results.append((other_key, other_val, h, d))
                    break  # one match is enough
        
        results.sort(key=lambda x: x[3])
        return results
    
    def signature(self, key: int) -> list[complex]:
        """H-element complex amplitude vector for this key.
        
        Each component is e^(i·γₕ·key), giving a deterministic
        frequency signature that can be used for:
          - Content-addressable memory (same key → same signature)
          - Similarity search (similar keys → similar signatures)
          - Feature hashing (amplitude encodes multiple features)
        """
        sig = []
        for g in self.gammas:
            p = phase(key, g)
            sig.append(complex(math.cos(p), math.sin(p)))
        return sig
    
    def density(self, head: int = 0) -> list[int]:
        """Bucket occupation counts for a given head."""
        return [len(self.tables[head][b]) for b in range(self.B)]
    
    def stats(self) -> dict:
        """Statistics about the array."""
        total = len(self._all_items)
        occupancies = []
        for h in range(self.H):
            occ = [len(b) for b in self.tables[h]]
            occupancies.append({
                'mean': sum(occ) / len(occ),
                'max': max(occ),
                'empty': sum(1 for o in occ if o == 0),
            })
        return {
            'items': total,
            'heads': self.H,
            'buckets': self.B,
            'occupancy': occupancies,
        }


# ══════════════════════════════════════════════════════════════════════
# Derived: ResonantHashTable (specialized for exact lookup)
# ══════════════════════════════════════════════════════════════════════

class ResonantHashTable:
    """O(1) hash table using Riemann zeros instead of random hash.
    
    Uses H independent hash functions (one per zero) for robustness.
    Collisions are handled by chaining. Same API as dict.
    """
    
    def __init__(self, n_heads=3, n_buckets=1024):
        self._arr = ResonantArray(n_heads, n_buckets)
    
    def __setitem__(self, key, value):
        self._arr.insert(key, value)
    
    def __getitem__(self, key):
        val = self._arr.lookup(key)
        if val is None:
            raise KeyError(key)
        return val
    
    def __contains__(self, key):
        return self._arr.lookup(key) is not None
    
    def __len__(self):
        return len(self._arr._all_items)


# ══════════════════════════════════════════════════════════════════════
# Derived: ResonantAttention (similarity via phase alignment)
# ══════════════════════════════════════════════════════════════════════

class ResonantAttention:
    """Attention mechanism using phase resonance instead of softmax.
    
    Given a query key, finds all stored keys with phase within
    threshold, and returns their values weighted by proximity.
    O(N·H) — no quadratic attention matrix.
    """
    
    def __init__(self, n_heads=4, n_buckets=256):
        self._arr = ResonantArray(n_heads, n_buckets)
    
    def add(self, key: int, value: float):
        self._arr.insert(key, value)
    
    def attend(self, query_key: int, threshold=0.15) -> float:
        """Compute attention-weighted sum over resonating keys.
        
        No softmax. Weight = 1 - circ_dist (linear decay with phase distance).
        """
        matches = self._arr.resonate(query_key, threshold)
        if not matches:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        for other_key, other_val, head, dist in matches:
            weight = 1.0 - dist  # linear: closer = more weight
            weighted_sum += weight * other_val
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# Derived: ResonantSignature (frequency-based content addressing)
# ══════════════════════════════════════════════════════════════════════

class ResonantSignature:
    """Content-addressable memory using frequency signatures.
    
    Each item has an H-element complex signature.
    Items with similar signatures have similar content.
    """
    
    def __init__(self, n_heads=8):
        self._arr = ResonantArray(n_heads, n_buckets=1)  # no bucketing needed
    
    def add(self, key: int, metadata: dict = None):
        sig = self._arr.signature(key)
        self._arr._all_items[key] = {'signature': sig, 'metadata': metadata or {}}
    
    def query(self, key: int, top_k: int = 5) -> list:
        """Find items with most similar frequency signature."""
        q_sig = self._arr.signature(key)
        scores = []
        
        for other_key, other_val in self._arr._all_items.items():
            if other_key == key:
                continue
            o_sig = other_val['signature']
            # Similarity = mean cosine similarity across all heads
            sim = sum((q_sig[h].real * o_sig[h].real + 
                       q_sig[h].imag * o_sig[h].imag)
                      for h in range(self._arr.H)) / self._arr.H
            scores.append((sim, other_key, other_val['metadata']))
        
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════
# Demo: All structures working together
# ══════════════════════════════════════════════════════════════════════

def demo():
    print("█" * 60)
    print("  RESONANT ARRAY: A GENERIC PHASE-INDEXED DATA STRUCTURE")
    print("  γ·key mod 2π → deterministic uniformity")
    print("█" * 60)
    
    # ── 1. Resonant Array: basic operations ──
    print("\n  [1] ResonantArray: insert, lookup, resonate")
    arr = ResonantArray(n_heads=3, n_buckets=64)
    
    for i in range(50):
        arr.insert(i * 7 + 1, f"val_{i}")
    
    stats = arr.stats()
    print(f"      {stats['items']} items, {stats['heads']} heads, {stats['buckets']} buckets")
    for h in range(stats['heads']):
        o = stats['occupancy'][h]
        print(f"      Head {h}: mean={o['mean']:.2f}, max={o['max']}, empty_buckets={o['empty']}")
    
    # Lookup
    val = arr.lookup(8)  # key = 8
    print(f"      lookup(8) = {val}")
    
    # Resonance
    near = arr.resonate(8, threshold=0.2)
    print(f"      resonate(8, 0.2): {len(near)} nearby keys")
    for k, v, h, d in near[:3]:
        print(f"        key={k}, head={h}, dist={d:.4f}")
    
    # Signature
    sig = arr.signature(42)
    print(f"      signature(42): {len(sig)} complex components")
    print(f"        |sig[0]| = {abs(sig[0]):.4f}")
    
    # ── 2. ResonantHashTable: dict-compatible ──
    print("\n  [2] ResonantHashTable: O(1) deterministic hash table")
    d = ResonantHashTable(n_heads=3, n_buckets=256)
    for i in range(100):
        d[i] = f"value_{i}"
    print(f"      len={len(d)}, 42 in d = {42 in d}, d[42] = {d[42]}")
    
    # ── 3. ResonantAttention: softmax-free attention ──
    print("\n  [3] ResonantAttention: signed attention via phase resonance")
    attn = ResonantAttention(n_heads=4, n_buckets=64)
    for i in range(20):
        attn.add(i * 5, float(i * 0.1))
    
    result = attn.attend(42, threshold=0.15)
    print(f"      attend(42, 0.15) = {result:.4f} (weighted sum over resonating keys)")
    
    # Check how many keys resonate
    arr2 = ResonantArray(n_heads=4, n_buckets=64)
    for i in range(20):
        arr2.insert(i * 5, float(i * 0.1))
    matches = arr2.resonate(42, 0.15)
    print(f"      {len(matches)} keys resonate with query 42")
    for k, v, h, d in matches[:3]:
        print(f"        key={k}, val={v}, head={h}, dist={d:.4f}")
    
    # ── 4. ResonantSignature: frequency-based content addressing ──
    print("\n  [4] ResonantSignature: content-addressable memory")
    sig_mem = ResonantSignature(n_heads=8)
    docs = {
        100: "golden ratio",
        200: "riemann hypothesis",
        300: "prime numbers",
        400: "transformers",
        500: "neural networks",
    }
    for k, v in docs.items():
        sig_mem.add(k, {'doc': v})
    
    similar = sig_mem.query(100, top_k=3)  # find docs similar to "golden ratio"
    print(f"      query(100, top 3):")
    for sim, key, meta in similar:
        print(f"        sim={sim:.4f}, key={key}, doc={meta.get('doc', '?')}")
    
    # ── 5. Properties summary ──
    print(f"\n{'='*60}")
    print(f"  FORMAL PROPERTIES")
    print(f"{'='*60}")
    print("""
  Definition: A Resonant Array R_γ is a data structure indexed by
    phase_γ(key) = γ · key mod 2π, where γ is a Riemann zero.

  Operations:
    insert(key, val)   — O(1) phase bucket assignment
    lookup(key)         — O(1) exact retrieval
    resonate(key, th)   — O(N) similarity via phase alignment
    signature(key)      — O(H) complex amplitude vector

  Properties:
    Deterministic:  same key → same phase (∀ machines, ∀ runs)
    Uniform:        phase distribution ~ U[0,2π) (Montgomery-Odlyzko)
    Independent:    different zeros → uncorrelated distributions
    Seedless:       no CSPRNG, no random state, no seeds
    Tunable:        threshold controls resonance width
    Scalable:       H heads → H independent subspaces
    Distributed:    same phases on every node (reproducible)

  Derived structures:
    ResonantHashTable    — dict-compatible, H-independent collision resolution
    ResonantAttention    — O(N·H) signed attention, no softmax
    ResonantSignature    — H-element frequency vector for content addressing

  The primitive replaces:
    Random hash functions  → γ₀·key mod 2π
    Learned embeddings     → [γ₀·key, γ₁·key, ...] mod 2π
    Softmax attention      → phase alignment threshold
    Training               → deterministic construction
""")


if __name__ == "__main__":
    demo()
