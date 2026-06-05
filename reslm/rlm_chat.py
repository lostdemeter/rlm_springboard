#!/usr/bin/env python3
"""
RLM Chat — interactive φ-resonant language model.

A Music Box approach: embeddings, attention, and generation
designed as a single system, tuned in lockstep.

Features:
  - φ-native embeddings from corpus (no GloVe, no training)
  - Multi-head resonant attention (signed, no softmax)
  - Autoregressive generation with attention diagnostics
  - Interactive chat interface with explainability

Type text to chat. /stats for diagnostics. /quit to exit.
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict

TWOPI = 2.0 * math.pi
PHI = (1 + 5 ** 0.5) / 2

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073,
          48.005151, 49.773832, 52.970322, 56.446248,
          59.347044, 60.831779, 65.112545, 67.079811]

def phase(key, gamma):
    return (gamma * key) % TWOPI

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def phi_zipf_mag(rank):
    return rank ** (-math.log(PHI))

# ══════════════════════════════════════════════════════════════════════
# 1. φ-Native Embeddings (pre-built or generated on the fly)
# ══════════════════════════════════════════════════════════════════════

def build_vocab_from_corpus(paths=None, vocab_size=500, max_tokens=50000):
    """Build vocabulary and co-occurrence from corpus."""
    if paths is None:
        paths = [
            '/home/thorin/termly_test4/riemann_constructive/alice.txt',
            '/home/thorin/termly_test4/riemann_constructive/moby.txt',
        ]
    
    words = []
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            words.extend(text.split()[:max_tokens])
    
    if len(words) < 100:
        words = ("the be to of and a in that have it i for not on with he as you "
                 "do at this but his by from they we her she or an will my one all "
                 "would there their what so up out if about who get which go me when "
                 "make can like time no just him know take people into year good "
                 "king queen man woman boy girl cat dog run walk love hate").split()
    
    freq = Counter(words)
    vocab = [w for w, _ in freq.most_common(vocab_size)]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, words


def build_cooc(vocab, w2i, words, window=5):
    """Build word co-occurrence AND bigram transition matrices.
    
    cooc[i][j] = how many times word i and word j appear within window
    trans[i][j] = how many times word j IMMEDIATELY FOLLOWS word i
    """
    V = len(vocab)
    cooc = [[0] * V for _ in range(V)]
    trans = [[0] * V for _ in range(V)]
    
    for pos, w in enumerate(words):
        if w not in w2i: continue
        i = w2i[w]
        # Co-occurrence (window-based)
        for ctx_pos in range(max(0, pos - window), min(len(words), pos + window + 1)):
            if ctx_pos == pos: continue
            cw = words[ctx_pos]
            if cw in w2i:
                cooc[i][w2i[cw]] += 1
        # Transition (bigram: what follows this word)
        if pos + 1 < len(words):
            nw = words[pos + 1]
            if nw in w2i:
                trans[i][w2i[nw]] += 1
    
    return cooc, trans


def cosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a)) + 1e-10
    nb = math.sqrt(sum(bi * bi for bi in b)) + 1e-10
    return dot / (na * nb)


def optimize_keys(vocab, w2i, cooc, n_iter=200):
    """Electrolysis: optimize keys for semantic phase alignment."""
    V = len(vocab)
    # Compute similarity for top word pairs
    top = min(100, V)
    pairs = []
    for i in range(top):
        for j in range(i + 1, top):
            sim = cosine(cooc[w2i[vocab[i]]], cooc[w2i[vocab[j]]])
            if abs(sim) > 0.01:
                pairs.append((vocab[i], vocab[j], sim))
    pairs.sort(key=lambda x: -abs(x[2]))
    
    keys = {w: float(i + 1) for i, w in enumerate(vocab)}
    best_k = dict(keys)
    best_s = -1e9
    
    for it in range(n_iter):
        idx = it * 3 % min(len(pairs), V)
        if idx >= len(pairs): continue
        a, b, target = pairs[idx]
        
        # Adjust keys toward target
        d = circ_dist(phase(keys[a] + 1, GAMMAS[0]), phase(keys[b] + 1, GAMMAS[0]))
        current = 1.0 - d
        delta = (target - current) * 0.5
        keys[a] = max(1, keys[a] + delta)
        keys[b] = max(1, keys[b] - delta)
        
        score = 0
        for pa, pb, pt in pairs[:100]:
            dd = circ_dist(phase(keys[pa] + 1, GAMMAS[0]), phase(keys[pb] + 1, GAMMAS[0]))
            score -= (1.0 - dd - pt) ** 2
        
        if score > best_s:
            best_s = score
            best_k = dict(keys)
    
    return best_k


class ResonantEmbedder:
    """φ-Zipf + Resonant Array deterministic embeddings."""
    def __init__(self, vocab, w2i, key_map=None, n_heads=16, n_dims=64):
        self.vocab = vocab
        self.w2i = w2i
        self.V = len(vocab)
        self.H = min(n_heads, n_dims // 2, len(GAMMAS))
        self.D = n_dims
        self.keys = {w: key_map.get(w, i + 1) if key_map else (i + 1) 
                     for i, w in enumerate(vocab)}
        self._cache = {}
    
    def embed(self, word):
        if word not in self.w2i:
            return None
        tid = self.w2i[word]
        if tid in self._cache:
            return self._cache[tid]
        
        rank = tid + 1
        key = self.keys.get(word, rank)
        mag_scale = phi_zipf_mag(rank)
        vec = []
        for h in range(self.H):
            g = GAMMAS[min(h, len(GAMMAS) - 1)]
            p = phase(key + 1, g)
            mag = (PHI ** (-h / 4.0)) * mag_scale
            vec.extend([mag * math.cos(p), mag * math.sin(p)])
        vec.extend([0.0] * (self.D - len(vec)))
        self._cache[tid] = vec
        return vec
    
    def embed_word(self, word):
        return self.embed(word)


# ══════════════════════════════════════════════════════════════════════
# 2. Resonant Attention
# ══════════════════════════════════════════════════════════════════════

class ResonantAttention:
    def __init__(self, n_heads=8, threshold=0.30):
        self.H = min(n_heads, len(GAMMAS))
        self.gammas = GAMMAS[:self.H]
        self.threshold = threshold
    
    def forward(self, embeds, token_ids, return_weights=False):
        T = len(embeds)
        if T == 0: return []
        D = len(embeds[0])
        Dh = D // self.H
        out = [list(e) for e in embeds]
        weights = [[[0.0] * T for _ in range(T)] for _ in range(self.H)]
        
        for h in range(self.H):
            g = self.gammas[h]
            q_phases = [phase(token_ids[i] + i + 1, g) for i in range(T)]
            start = h * Dh
            
            for i in range(T):
                total_w = 0.0
                wsum = [0.0] * Dh
                for j in range(i + 1):
                    d = circ_dist(q_phases[i], q_phases[j])
                    if d < self.threshold:
                        w = 1.0 - d / self.threshold
                        weights[h][i][j] = w
                        total_w += w
                        for di in range(Dh):
                            if start + di < D:
                                wsum[di] += w * embeds[j][start + di]
                
                if total_w > 0:
                    for di in range(Dh):
                        if start + di < D:
                            out[i][start + di] = wsum[di] / total_w
        
        if return_weights:
            return out, weights
        return out


# ══════════════════════════════════════════════════════════════════════
# 3. Chat Model
# ══════════════════════════════════════════════════════════════════════

class RLMChat:
    def __init__(self, vocab, w2i, embedder, trans_matrix=None):
        self.vocab = vocab
        self.w2i = w2i
        self.emb = embedder
        self.attn = ResonantAttention(n_heads=8, threshold=0.30)
        self.V = len(vocab)
        self.trans = trans_matrix
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = [w for w in text.split() if w]
        ids = [self.w2i.get(w, 0) for w in tokens if w in self.w2i]
        return tokens, ids
    
    def predict_next(self, token_ids, prompt_tokens, return_analysis=False):
        T = len(token_ids)
        if T == 0: return 0, {}
        
        # Embed
        embeds = []
        for tid in token_ids:
            e = self.emb.embed(self.vocab[tid] if tid < self.V else self.vocab[0])
            embeds.append(e)
        
        # Attention
        attended, weights = self.attn.forward(embeds, token_ids, return_weights=True)
        
        # Output: only consider CONTENT/DETAIL band words (not ROUTE/function words)
        # φ-Zipf pole separation: function words all point same direction,
        # content words each have unique φ-direction → better for generation
        final = attended[-1]
        D = len(final)
        scores = {}
        for tid in range(min(self.V, len(self.vocab))):
            w = self.vocab[tid]
            rank = tid + 1
            # Skip ROUTE band words (function words — common-word pole)
            band_frac = math.log1p(rank) / math.log1p(self.V)
            if band_frac < 0.20:  # ROUTE band
                continue
            if band_frac > 0.55:  # rare DETAIL, low signal
                continue
            
            e = self.emb.embed(w)
            if e is None: continue
            f_norm = math.sqrt(sum(v*v for v in final)) + 1e-10
            e_norm = math.sqrt(sum(v*v for v in e)) + 1e-10
            s = sum(final[d] * e[d] for d in range(min(D, len(e)))) / (f_norm * e_norm)
            scores[tid] = s
        
        # Boost content words that appeared in the prompt (context relevance)
        for tid in range(self.V):
            w = self.vocab[tid]
            if w in prompt_tokens:
                scores[tid] = scores.get(tid, 0) + 0.3
        
        # Bigram transition bias: words that ACTUALLY follow the previous token
        if self.trans and len(token_ids) >= 1:
            prev_tid = token_ids[-1]
            if prev_tid < self.V:
                row = self.trans[prev_tid]
                row_total = sum(row)
                if row_total > 0:
                    for tid in range(self.V):
                        if row[tid] > 0:
                            prob = row[tid] / row_total
                            scores[tid] = scores.get(tid, 0) + prob * 0.8
        
        # Anti-repetition: strongly penalize words already in recent output.
        # This prevents the "as as as as" degenerate pattern.
        recent = token_ids[-4:] if len(token_ids) >= 4 else token_ids
        for tid in recent:
            if tid in scores:
                scores[tid] -= 0.5  # penalty for recent appearance
        
        # Top-k with recency tiebreak
        sorted_s = sorted(scores.items(), key=lambda x: -x[1])
        top_k = 5
        top = sorted_s[:top_k]
        best_score = top[0][1]
        
        # Prefer: higher score, then most recent in prompt, then highest ID
        recency = {t: -1 for t in range(self.V)}
        for pos, t in enumerate(token_ids):
            recency[t] = pos
        
        candidates = [t for t, s in top if s >= best_score - 0.01]
        best = max(candidates, key=lambda t: (scores[t], recency[t], t))
        
        if return_analysis:
            return best, scores, weights, attended
        return best, scores
    
    def generate(self, prompt, max_tokens=15):
        tokens, ids = self.tokenize(prompt)
        if not ids:
            return "", {}
        
        prompt_words = [w for w in tokens if w in self.w2i]
        result = list(prompt_words)
        result_ids = list(ids)
        
        for _ in range(max_tokens):
            tid, scores = self.predict_next(result_ids, tokens)
            word = self.vocab[tid]
            result.append(word)
            result_ids.append(tid)
        
        return ' '.join(result), scores
    
    def generate_with_analysis(self, prompt, max_tokens=10):
        tokens, ids = self.tokenize(prompt)
        if not ids:
            return "?", {}
        
        prompt_words = [w for w in tokens if w in self.w2i]
        result = list(prompt_words)
        result_ids = list(ids)
        
        print(f"  Tokens: {result}")
        print(f"  Embedding dim: {self.emb.D}, Heads: {self.attn.H}")
        
        for step in range(max_tokens):
            tid, scores, wts, attended = self.predict_next(
                result_ids, tokens, return_analysis=True)
            word = self.vocab[tid]
            result.append(word)
            result_ids.append(tid)
            
            # Show top candidates
            sorted_s = sorted(scores.items(), key=lambda x: -x[1])[:5]
            top_str = ' | '.join(f"{self.vocab[t]}={s:.1f}" for t, s in sorted_s)
            
            # Show attention for last position
            attn_summary = ""
            if wts and len(result_ids) > 1:
                for h in range(min(2, self.attn.H)):
                    last_wts = wts[h][-1]
                    strong = [(j, w) for j, w in enumerate(last_wts) if w > 0.3]
                    if strong:
                        words = [prompt_words[j] if j < len(prompt_words) else result[j-len(prompt_words)] 
                                for j, _ in strong[:3]]
                        attn_summary += f"  h{h}→{','.join(words)}"
            
            print(f"  → {word:>10s}  [{top_str}]  {attn_summary}")
        
        return ' '.join(result), scores


# ══════════════════════════════════════════════════════════════════════
# 4. Main Chat Loop
# ══════════════════════════════════════════════════════════════════════

def main():
    print("█" * 60)
    print("  RLM CHAT — φ-Resonant Language Model")
    print("  Embeddings + Attention + Generation — one system")
    print("█" * 60)
    
    # Build model
    print("\n  Building φ-native embeddings from corpus...")
    vocab, w2i, words = build_vocab_from_corpus()
    cooc, trans = build_cooc(vocab, w2i, words)
    
    print(f"  Vocabulary: {len(vocab)} words from {len(words)} tokens")
    print(f"  Optimizing keys (electrolysis)...")
    key_map = optimize_keys(vocab, w2i, cooc, n_iter=150)
    
    embedder = ResonantEmbedder(vocab, w2i, key_map)
    chat = RLMChat(vocab, w2i, embedder, trans)
    
    print(f"  Ready. Type text to chat, /help for commands.\n")
    
    while True:
        try:
            user = input("  YOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        
        if not user:
            continue
        
        if user.startswith('/'):
            cmd = user[1:].strip().lower()
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                print("  Commands: /stats /embed <word> /similar <word> /quit")
            elif cmd == 'stats':
                print(f"  Vocab: {len(vocab)}  Embeddings: {embedder.D}D  Heads: {embedder.H}")
                print(f"  Optimized keys: {len([k for k in key_map.values() if k != int(k)])} adjusted")
            elif cmd.startswith('embed '):
                word = cmd[6:].strip()
                e = embedder.embed_word(word)
                if e:
                    norm = math.sqrt(sum(v*v for v in e))
                    print(f"  {word}: |e|={norm:.4f}, [{', '.join(f'{v:+.2f}' for v in e[:8])}...]")
                else:
                    print(f"  '{word}' not in vocabulary")
            elif cmd.startswith('similar '):
                word = cmd[8:].strip()
                e1 = embedder.embed_word(word)
                if not e1:
                    print(f"  '{word}' not in vocabulary")
                    continue
                similarities = []
                for w in vocab[:100]:
                    if w == word: continue
                    e2 = embedder.embed_word(w)
                    if e2:
                        similarities.append((w, cosine(e1, e2)))
                similarities.sort(key=lambda x: -x[1])
                for w, s in similarities[:8]:
                    print(f"    {w:>12s}: {s:+.3f}")
            else:
                print(f"  Unknown command: {cmd}")
        else:
            # Generate with analysis for first message, simple for subsequent
            result, scores = chat.generate_with_analysis(user, max_tokens=10)
            print(f"  RLM> {result}\n")


if __name__ == "__main__":
    main()
