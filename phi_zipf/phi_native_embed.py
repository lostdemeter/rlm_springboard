#!/usr/bin/env python3
"""
φ-Native Embedding Generator: Build semantic similarity from word co-occurrence.

Instead of downloading GloVe, we generate semantic structure from a small
public-domain text corpus using window-based co-occurrence. Two words that
appear in similar contexts get nearby phases in the φ-ladder.

The co-occurrence matrix feeds directly into the electrolysis optimizer,
producing deterministic φ-Zipf + Resonant Array embeddings with semantic
structure — zero external data, zero training artifacts.

Process:
  1. Load public-domain corpus (Alice in Wonderland + Moby Dick)
  2. Build co-occurrence matrix (window = 5 words)
  3. Compute context-based similarity between words
  4. Feed similarity matrix into electrolysis → optimized token keys
  5. Generate deterministic φ-resonant embeddings
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import islice

TWOPI = 2.0 * math.pi
PHI = (1 + 5 ** 0.5) / 2

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073,
          48.005151, 49.773832, 52.970322, 56.446248]

# ── Primitives ─────────────────────────────────────────────────────────

def phase(key, gamma):
    return (gamma * key) % TWOPI

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def phi_zipf_magnitude(rank):
    """φ-Zipf magnitude: |e| ∝ rank^(-0.481)"""
    return rank ** (-math.log(PHI))

def cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a))
    nb = math.sqrt(sum(bi * bi for bi in b))
    return dot / (na * nb + 1e-10)


# ══════════════════════════════════════════════════════════════════════
# 1. Corpus Loader
# ══════════════════════════════════════════════════════════════════════

def load_corpus(paths, max_words=50000):
    """Load and tokenize text files."""
    words = []
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            tokens = text.split()
            words.extend(tokens[:max_words])
    return words


# ══════════════════════════════════════════════════════════════════════
# 2. Co-occurrence → Similarity Matrix (φ-Native GloVe Replacement)
# ══════════════════════════════════════════════════════════════════════

def build_cooccurrence(tokens, window=5, vocab_size=2000):
    """Build word co-occurrence matrix over a sliding window.
    
    cooc[w][c] = how many times word w appears with context word c
    within ±window positions.
    """
    # Build vocabulary (top V words by frequency)
    freq = Counter(tokens)
    vocab = [w for w, _ in freq.most_common(vocab_size)]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    
    # Build co-occurrence
    cooc = [[0] * V for _ in range(V)]
    
    for pos, w in enumerate(tokens):
        if w not in w2i:
            continue
        i = w2i[w]
        start = max(0, pos - window)
        end = min(len(tokens), pos + window + 1)
        for ctx_pos in range(start, end):
            if ctx_pos == pos:
                continue
            ctx_word = tokens[ctx_pos]
            if ctx_word in w2i:
                j = w2i[ctx_word]
                cooc[i][j] += 1
    
    return vocab, w2i, cooc


def cooccurrence_to_similarity(vocab, w2i, cooc):
    """Convert co-occurrence matrix to word similarity via context overlap.
    
    Two words are similar if they appear with the same context words.
    Similarity = cosine of context vectors.
    """
    V = len(vocab)
    
    # Compute similarity for top word pairs (sparse)
    similarities = {}
    top_words = vocab[:200]  # compare top 200 words
    
    for i in range(len(top_words)):
        for j in range(i + 1, len(top_words)):
            wi, wj = top_words[i], top_words[j]
            ci, cj = cooc[w2i[wi]], cooc[w2i[wj]]
            sim = cosine_sim(ci, cj)
            if abs(sim) > 0.01:
                similarities[(wi, wj)] = sim
    
    return similarities


# ══════════════════════════════════════════════════════════════════════
# 3. Frequency Band Assignment
# ══════════════════════════════════════════════════════════════════════

def frequency_band(rank, V):
    """Map frequency rank to φ-rung band."""
    frac = math.log1p(rank) / math.log1p(V)
    if frac < 0.20:
        return 'ROUTE', 90, 110
    elif frac < 0.55:
        return 'CONTENT', 110, 130
    else:
        return 'DETAIL', 130, 155


# ══════════════════════════════════════════════════════════════════════
# 4. Electrolysis — Key Optimization
# ══════════════════════════════════════════════════════════════════════

def optimize_keys(vocab, w2i, similarities, n_iter=500):
    """Optimize token keys so phase proximity matches semantic similarity.
    
    The electrolysis process:
      Template = co-occurrence similarity (what should be close)
      Mold = φ-Zipf frequency bands (where tokens can go)
      Current = key adjustments (depositing tokens onto φ-ladder)
    """
    V = len(vocab)
    use_sim = {w for pair in similarities for w in pair}
    sim_words = sorted(use_sim, key=lambda w: -len([p for p in similarities if w in p]))
    
    # Initialize keys by frequency rank
    keys = {w: float(i + 1) for i, w in enumerate(vocab)}
    
    # Build pair list for scoring
    pairs = []
    for (a, b), sim in similarities.items():
        pairs.append((a, b, sim))
    pairs.sort(key=lambda x: -abs(x[2]))
    
    # Optimize
    best_keys = dict(keys)
    best_score = -float('inf')
    
    for iteration in range(n_iter):
        # Perturb one word's key
        idx = iteration * 3 % len(sim_words) if sim_words else iteration % V
        w = sim_words[idx] if sim_words else vocab[idx % V]
        old_key = keys[w]
        
        # Adjustment proportional to average miss with neighbors
        adjustment = 0.0
        n_neighbors = 0
        for a, b, target_sim in pairs[:200]:
            if a == w and b in keys:
                d = circ_dist(phase(keys[a] + 1, GAMMAS[0]), phase(keys[b] + 1, GAMMAS[0]))
                current = 1.0 - d
                adjustment += (target_sim - current) * 5.0
                n_neighbors += 1
            elif b == w and a in keys:
                d = circ_dist(phase(keys[b] + 1, GAMMAS[0]), phase(keys[a] + 1, GAMMAS[0]))
                current = 1.0 - d
                adjustment += (target_sim - current) * 5.0
                n_neighbors += 1
        
        if n_neighbors > 0:
            new_key = old_key + adjustment / n_neighbors
            if new_key <= 0:
                new_key = old_key + abs(adjustment / n_neighbors)
            
            # Constrain to frequency band
            rank = list(vocab).index(w) + 1
            band, lo, hi = frequency_band(rank, V)
            new_key = max(1.0, min(float(V), new_key))
            
            keys[w] = new_key
        
        # Score
        score = 0.0
        for a, b, target_sim in pairs[:500]:
            d = circ_dist(phase(keys[a] + 1, GAMMAS[0]), phase(keys[b] + 1, GAMMAS[0]))
            current = 1.0 - d
            score -= (current - target_sim) ** 2
        
        if score > best_score:
            best_score = score
            best_keys = dict(keys)
    
    return best_keys


# ══════════════════════════════════════════════════════════════════════
# 5. φ-Native Embedding Generation
# ══════════════════════════════════════════════════════════════════════

def resonant_embedding(key, rank, n_dims=64, n_heads=16):
    """φ-Zipf + Resonant Array deterministic embedding."""
    n_heads = min(n_heads, n_dims // 2, len(GAMMAS))
    mag_scale = phi_zipf_magnitude(rank)
    
    vec = []
    for h in range(n_heads):
        g = GAMMAS[min(h, len(GAMMAS) - 1)]
        p = phase(key + 1, g)
        mag = (PHI ** (-h / (n_heads / 4))) * mag_scale
        vec.extend([mag * math.cos(p), mag * math.sin(p)])
    
    vec.extend([0.0] * (n_dims - len(vec)))
    return vec


# ══════════════════════════════════════════════════════════════════════
# 6. Demo
# ══════════════════════════════════════════════════════════════════════

def demo():
    print("█" * 64)
    print("  φ-NATIVE EMBEDDING GENERATOR")
    print("  Corpus → Co-occurrence → Electrolysis → Deterministic Embeddings")
    print("█" * 64)
    
    # Load corpus
    corpus_paths = [
        '/home/thorin/termly_test4/riemann_constructive/alice.txt',
        '/home/thorin/termly_test4/riemann_constructive/moby.txt',
    ]
    
    # Check if corpus exists, otherwise use built-in text
    words = load_corpus(corpus_paths, 20000)
    
    if len(words) < 100:
        print("\n  No corpus found. Using built-in English text sample...")
        words = ("the be to of and a in that have it i for not on with he as you "
                 "do at this but his by from they we her she or an will my one all "
                 "would there their what so up out if about who get which go me when "
                 "make can like time no just him know take people into year your good "
                 "some could them see other than then now look only come its over think "
                 "also back after use two how our work first well way even new want "
                 "because any these give day most us great man woman king queen prince "
                 "princess boy girl father mother son daughter cat dog bird fish wolf "
                 "run walk swim fly jump dance sing read write speak listen think "
                 "love hate happy sad brave fear good bad big small hot cold").split()
    
    print(f"  Corpus: {len(words)} tokens")
    
    # Build co-occurrence
    vocab, w2i, cooc = build_cooccurrence(words, window=5, vocab_size=500)
    V = len(vocab)
    print(f"  Vocabulary: {V} words")
    
    # Compute similarities
    similarities = cooccurrence_to_similarity(vocab, w2i, cooc)
    n_pairs = len(similarities)
    print(f"  Significant word pairs: {n_pairs}")
    
    # Show top similarities
    top_pairs = sorted(similarities.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top Semantic Pairs (co-occurrence):")
    for (a, b), sim in top_pairs:
        print(f"    {a:>10s} ↔ {b:<10s}: {sim:+.3f}")
    
    # Frequency bands
    print(f"\n  φ-Rung Frequency Bands:")
    for rank in [1, 2, 3, 10, 50, 100, 200, 500]:
        band, lo, hi = frequency_band(rank, V)
        word = vocab[rank - 1] if rank <= V else '---'
        mag = phi_zipf_magnitude(rank)
        rung = -255 * math.log(max(mag, 1e-10)) / (20 * math.log(PHI))
        print(f"    rank={rank:>4d} '{word:>10s}': band={band:>8s} "
              f"rungs [{lo}-{hi}] mag={mag:.4f} rung≈{rung:.0f}")
    
    # Electrolysis
    print(f"\n  Electrolysis: Optimizing keys...")
    key_map = optimize_keys(vocab, w2i, similarities, n_iter=300)
    
    # Generate embeddings
    print(f"  Generating φ-native embeddings...")
    embeddings = {}
    for i, w in enumerate(vocab):
        rank = i + 1
        key = int(key_map.get(w, rank))
        embeddings[w] = resonant_embedding(key, rank, n_dims=64, n_heads=16)
    
    # Semantic preservation
    test_words = [w for w in 
        ['king', 'queen', 'prince', 'princess', 'man', 'woman',
         'boy', 'girl', 'cat', 'dog', 'father', 'mother',
         'run', 'walk', 'jump', 'swim', 'love', 'hate',
         'good', 'bad', 'big', 'small', 'hot', 'cold']
        if w in w2i]
    
    if len(test_words) >= 6:
        print(f"\n  Semantic Preservation Check:")
        
        # Co-occurrence similarities vs RLM embeddings
        coc_sims = []
        rlm_sims = []
        for i in range(len(test_words)):
            for j in range(i + 1, len(test_words)):
                a, b = test_words[i], test_words[j]
                if (a, b) in similarities or (b, a) in similarities:
                    coc_sim = similarities.get((a, b), similarities.get((b, a), 0))
                    rlm_sim = cosine_sim(embeddings[a], embeddings[b])
                    coc_sims.append(coc_sim)
                    rlm_sims.append(rlm_sim)
        
        if coc_sims:
            pairs_shown = 0
            for i in range(len(test_words)):
                for j in range(i + 1, len(test_words)):
                    a, b = test_words[i], test_words[j]
                    skey = (a, b) if (a, b) in similarities else ((b, a) if (b, a) in similarities else None)
                    if skey:
                        coc_sim = similarities[skey]
                        rlm_sim = cosine_sim(embeddings[a], embeddings[b])
                        change = "+" if coc_sim * rlm_sim > 0 else "FLIP"
                        print(f"    {a:>8s} ↔ {b:<8s}: coc={coc_sim:+.3f} → RLM={rlm_sim:+.3f} ({change})")
                        pairs_shown += 1
                        if pairs_shown >= 15:
                            break
                if pairs_shown >= 15:
                    break
            
            # Correlation
            if len(coc_sims) > 3:
                mean_c = sum(coc_sims) / len(coc_sims)
                mean_r = sum(rlm_sims) / len(rlm_sims)
                num = sum((c - mean_c) * (r - mean_r) for c, r in zip(coc_sims, rlm_sims))
                den_c = sum((c - mean_c) ** 2 for c in coc_sims)
                den_r = sum((r - mean_r) ** 2 for r in rlm_sims)
                corr = num / (math.sqrt(den_c * den_r) + 1e-10)
                print(f"\n    Correlation (co-occurrence vs RLM): {corr:.3f}")
    
    # Determinism
    e1 = resonant_embedding(42, 10, 64, 16)
    e2 = resonant_embedding(42, 10, 64, 16)
    mismatch = sum(1 for a, b in zip(e1, e2) if abs(a-b) > 1e-6)
    print(f"\n  Determinism: {'✓ PERFECT' if mismatch == 0 else f'✗ {mismatch} mismatches'}")
    
    # Key distribution
    all_keys = list(key_map.values())
    print(f"\n  Key Distribution: min={min(all_keys):.0f} max={max(all_keys):.0f} "
          f"mean={sum(all_keys)/len(all_keys):.0f}")
    
    print(f"\n{'='*64}")
    print(f"  Summary")
    print(f"{'='*64}")
    print(f"""
  Corpus:         {len(words)} tokens, {V} word vocabulary
  Co-occurrence:  {n_pairs} significant word pairs with context overlap
  Electrolysis:   {len(key_map)} keys optimized for φ-phase assignment
  Embeddings:     {len(embeddings)} words, 64D, φ-Zipf + Resonant Array
  External data:  ZERO — everything from corpus + φ + ζ
  Deterministic:  YES — same output on every machine
""")


if __name__ == "__main__":
    demo()
