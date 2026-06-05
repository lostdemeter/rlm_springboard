#!/usr/bin/env python3
"""
Rigorous RLM Tuning — embedded model, automated parameter discovery.

The RLM is embedded directly so parameters flow through instantly.
Ollama evaluates each configuration and generates reference completions
that feed back into the bigram transition matrix — the model improves
with each cycle.

Architecture:
  1. RLM generates text with current parameters
  2. Ollama scores coherence (0-10)
  3. Ollama generates reference completions for training signal
  4. Reference transitions are injected into the RLM's bigram matrix
  5. Parameters are adjusted based on score feedback
  6. Repeat — each cycle tightens the Music Box

Run:  python3 rigorous_tune.py [--iterations 100] [--model devstral-small-2:24b]
"""

import math
import os
import re
import sys
import subprocess
import random
import argparse
from collections import Counter, defaultdict

TWOPI = 2.0 * math.pi
PHI = (1 + 5 ** 0.5) / 2

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073,
          48.005151, 49.773832, 52.970322, 56.446248]


# ══════════════════════════════════════════════════════════════════════
# RLM Core — embedded (no subprocess)
# ══════════════════════════════════════════════════════════════════════

def phase(key, gamma):
    return (gamma * key) % TWOPI

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def phi_zipf_mag(rank):
    return rank ** (-math.log(PHI))

def cosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a)) + 1e-10
    nb = math.sqrt(sum(bi * bi for bi in b)) + 1e-10
    return dot / (na * nb)


class FastRLM:
    """Lightweight embedded RLM — no PyTorch, no training, fully tunable."""
    
    def __init__(self, vocab_size=500):
        self.V = vocab_size
        self.H = 8
        self.D = 64
        self.gammas = GAMMAS[:self.H]
        
        # Tunable parameters (Music Box knobs)
        self.attn_threshold = 0.30
        self.route_band = 0.20
        self.content_band = 0.55
        self.trans_weight = 0.8
        self.anti_repeat = 0.5
        self.prompt_boost = 0.3
        self.n_heads = 8
        
        # Data: set by init_from_corpus
        self.vocab = []
        self.w2i = {}
        self.trans = None
        self.keys = {}
        self._emb_cache = {}
    
    def init_from_corpus(self, max_vocab=500):
        """Load corpus and build vocabulary, transitions, embeddings."""
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
                words.extend(text.split()[:30000])
        
        if len(words) < 100:
            words = ("the be to of and a in that have it i for not on with he as you "
                     "do at this but his by from they we her she or an will my one all "
                     "would there their what so up out if about who get which go me when "
                     "make can like time no just him know take people into year good "
                     "some could them see other than then now look only come its over "
                     "think also back after use two how our work first well way even new "
                     "want because any these give day most us great man woman king queen "
                     "run walk swim fly jump dance love hate happy sad brave fear "
                     "good bad big small hot cold boy girl cat dog bird fish").split()
        
        freq = Counter(words)
        self.vocab = [w for w, _ in freq.most_common(max_vocab)]
        self.V = len(self.vocab)
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        
        # Build transition matrices at multiple n-gram levels
        self.trans = [[0] * self.V for _ in range(self.V)]       # bigram
        self.trigram = {}  # (w_i, w_j) → {w_k: count}
        self.quadgram = {}  # (w_i, w_j, w_k) → {w_l: count}
        
        for pos in range(len(words) - 1):
            w = words[pos]
            if w not in self.w2i: continue
            # Bigram
            nw = words[pos + 1]
            if nw in self.w2i:
                self.trans[self.w2i[w]][self.w2i[nw]] += 1
            # Trigram
            if pos >= 1 and pos + 1 < len(words):
                w_prev = words[pos - 1]
                if w_prev in self.w2i and w in self.w2i and nw in self.w2i:
                    key = (self.w2i[w_prev], self.w2i[w])
                    if key not in self.trigram:
                        self.trigram[key] = Counter()
                    self.trigram[key][self.w2i[nw]] += 1
            # Quadgram
            if pos >= 2 and pos + 1 < len(words):
                w_pp = words[pos - 2]
                w_p = words[pos - 1]
                if w_pp in self.w2i and w_p in self.w2i and w in self.w2i and nw in self.w2i:
                    key = (self.w2i[w_pp], self.w2i[w_p], self.w2i[w])
                    if key not in self.quadgram:
                        self.quadgram[key] = Counter()
                    self.quadgram[key][self.w2i[nw]] += 1
        
        # Build co-occurrence matrix
        cooc = [[0] * self.V for _ in range(self.V)]
        for pos, w in enumerate(words):
            if w not in self.w2i: continue
            i = self.w2i[w]
            for ctx_pos in range(max(0, pos - 5), min(len(words), pos + 6)):
                if ctx_pos == pos: continue
                cw = words[ctx_pos]
                if cw in self.w2i:
                    cooc[i][self.w2i[cw]] += 1
        
        # Optimize keys via electrolysis
        self.keys = {w: float(i + 1) for i, w in enumerate(self.vocab)}
        top = min(100, self.V)
        pairs = []
        for i in range(top):
            for j in range(i + 1, top):
                sim = cosine(cooc[i], cooc[j])
                if abs(sim) > 0.01:
                    pairs.append((self.vocab[i], self.vocab[j], sim))
        pairs.sort(key=lambda x: -abs(x[2]))
        
        best_k = dict(self.keys)
        best_s = -1e9
        for it in range(200):
            idx = it * 3 % min(len(pairs), self.V)
            if idx >= len(pairs): continue
            a, b, target = pairs[idx]
            d = circ_dist(phase(self.keys[a]+1, GAMMAS[0]), phase(self.keys[b]+1, GAMMAS[0]))
            current = 1.0 - d
            delta = (target - current) * 0.5
            self.keys[a] = max(1, self.keys[a] + delta)
            self.keys[b] = max(1, self.keys[b] - delta)
            score = 0
            for pa, pb, pt in pairs[:100]:
                dd = circ_dist(phase(self.keys[pa]+1, GAMMAS[0]), phase(self.keys[pb]+1, GAMMAS[0]))
                score -= (1.0 - dd - pt) ** 2
            if score > best_s:
                best_s = score
                best_k = dict(self.keys)
        self.keys = best_k
        
        print(f"  Corpus: {len(words)} tokens, {self.V} vocab, {len(pairs)} electrolysis pairs")
    
    def embed(self, word):
        if word not in self.w2i:
            return None
        tid = self.w2i[word]
        if tid in self._emb_cache:
            return self._emb_cache[tid]
        
        rank = tid + 1
        key = self.keys.get(word, rank)
        mag_scale = phi_zipf_mag(rank)
        vec = []
        for h in range(self.H):
            g = GAMMAS[min(h, len(GAMMAS)-1)]
            p = phase(key + 1, g)
            mag = (PHI ** (-h / 4.0)) * mag_scale
            vec.extend([mag * math.cos(p), mag * math.sin(p)])
        vec.extend([0.0] * (self.D - len(vec)))
        self._emb_cache[tid] = vec
        return vec
    
    def generate(self, prompt, max_tokens=10):
        """Full forward pass: embed → attend → score → decode."""
        # Tokenize
        text = prompt.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        prompt_words = [w for w in text.split() if w in self.w2i]
        if not prompt_words:
            result = list(prompt_words)
            token_ids = []
        else:
            result = list(prompt_words)
            token_ids = [self.w2i[w] for w in prompt_words]
        
        for _ in range(max_tokens):
            if not token_ids:
                break
            
            T = len(token_ids)
            embs = [self.embed(self.vocab[tid]) for tid in token_ids]
            
            # ── Attention ──
            attended = [list(e) for e in embs]
            for h in range(min(self.n_heads, self.H)):
                g = GAMMAS[h]
                q_phases = [phase(token_ids[i] + i + 1, g) for i in range(T)]
                start = h * (self.D // self.H)
                Dh = self.D // self.H
                
                for i in range(T):
                    total_w = 0.0
                    wsum = [0.0] * Dh
                    for j in range(i + 1):
                        d = circ_dist(q_phases[i], q_phases[j])
                        if d < self.attn_threshold:
                            w = 1.0 - d / self.attn_threshold
                            total_w += w
                            for di in range(Dh):
                                si = start + di
                                if si < self.D:
                                    wsum[di] += w * embs[j][si]
                    if total_w > 0:
                        for di in range(Dh):
                            si = start + di
                            if si < self.D:
                                attended[i][si] = wsum[di] / total_w
            
            # ── Scoring ──
            final = attended[-1]
            D = len(final)
            scores = {}
            
            for tid in range(self.V):
                band_frac = math.log1p(tid + 1) / math.log1p(self.V)
                if band_frac < self.route_band:
                    continue  # skip function words
                
                e = self.embed(self.vocab[tid])
                if e is None: continue
                f_norm = math.sqrt(sum(v*v for v in final)) + 1e-10
                e_norm = math.sqrt(sum(v*v for v in e)) + 1e-10
                scores[tid] = sum(final[d] * e[d] for d in range(min(D, len(e)))) / (f_norm * e_norm)
            
            # Context boost
            for tid in range(self.V):
                w = self.vocab[tid]
                if w in prompt_words:
                    scores[tid] = scores.get(tid, 0) + self.prompt_boost
            
            # N-gram transitions: nested context at multiple levels
            # Bigram: what follows the previous word
            if self.trans and token_ids:
                prev = token_ids[-1]
                if prev < self.V:
                    row = self.trans[prev]
                    rt = sum(row)
                    if rt > 0:
                        for tid in range(self.V):
                            if row[tid] > 0:
                                scores[tid] = scores.get(tid, 0) + (row[tid] / rt) * self.trans_weight
            # Trigram: what follows the previous TWO words (stronger signal)
            if self.trigram and len(token_ids) >= 2:
                key = (token_ids[-2], token_ids[-1])
                if key in self.trigram:
                    counts = self.trigram[key]
                    ct = sum(counts.values())
                    if ct > 0:
                        for tid, cnt in counts.items():
                            if tid < self.V:
                                scores[tid] = scores.get(tid, 0) + (cnt / ct) * self.trans_weight * 1.5
            # Quadgram: what follows the previous THREE words (strongest signal)
            if self.quadgram and len(token_ids) >= 3:
                key = (token_ids[-3], token_ids[-2], token_ids[-1])
                if key in self.quadgram:
                    counts = self.quadgram[key]
                    ct = sum(counts.values())
                    if ct > 0:
                        for tid, cnt in counts.items():
                            if tid < self.V:
                                scores[tid] = scores.get(tid, 0) + (cnt / ct) * self.trans_weight * 2.0
            
            # Anti-repetition
            recent = token_ids[-3:] if len(token_ids) >= 3 else token_ids
            for tid in recent:
                if tid in scores:
                    scores[tid] -= self.anti_repeat
            
            # Select best
            if not scores:
                break
            top = sorted(scores.items(), key=lambda x: -x[1])
            best_score = top[0][1]
            candidates = [t for t, s in top[:5] if s >= best_score - 0.01]
            recency = {t: -1 for t in range(self.V)}
            for pos, t in enumerate(token_ids):
                recency[t] = pos
            best = max(candidates, key=lambda t: (scores[t], recency[t], t))
            
            result.append(self.vocab[best])
            token_ids.append(best)
        
        return ' '.join(result), token_ids
    
    def inject_reference(self, reference_text):
        """Feed a reference completion into the transition matrix.
        
        Each bigram in the reference teaches the model what words
        should follow each other in coherent text.
        Handles OOV words by adding them to vocabulary dynamically.
        """
        words = reference_text.lower().split()
        bigrams_injected = 0
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            # Add unknown words to vocabulary
            if a not in self.w2i:
                self.w2i[a] = self.V
                self.vocab.append(a)
                self.V += 1
                # Expand transition matrix
                self.trans.append([0] * self.V)
                for row in self.trans[:-1]:
                    row.append(0)
            if b not in self.w2i:
                self.w2i[b] = self.V
                self.vocab.append(b)
                self.V += 1
                self.trans.append([0] * self.V)
                for row in self.trans[:-1]:
                    row.append(0)
            
            ia, ib = self.w2i[a], self.w2i[b]
            self.trans[ia][ib] += 3  # boost reference transitions
            bigrams_injected += 1
            
            # Also inject into trigram/quadgram if we have context
            if i >= 1:
                pa = words[i - 1]
                if pa in self.w2i:
                    tkey = (self.w2i[pa], ia)
                    if tkey not in self.trigram:
                        self.trigram[tkey] = Counter()
                    self.trigram[tkey][ib] += 3
            if i >= 2:
                ppa = words[i - 2]
                pa = words[i - 1]
                if ppa in self.w2i and pa in self.w2i:
                    qkey = (self.w2i[ppa], self.w2i[pa], ia)
                    if qkey not in self.quadgram:
                        self.quadgram[qkey] = Counter()
                    self.quadgram[qkey][ib] += 3
        return bigrams_injected


# ══════════════════════════════════════════════════════════════════════
# Ollama Interface
# ══════════════════════════════════════════════════════════════════════

def ask_ollama(prompt, model, system=None):
    try:
        inp = f"{system}\n\n{prompt}" if system else prompt
        result = subprocess.run(
            ["ollama", "run", model], input=inp,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', result.stdout).strip()
    except:
        pass
    return None

def score_coherence(prompt, completion, model):
    resp = ask_ollama(
        f'Rate how well this completion continues the prompt. Reply with ONLY a number 0-10.\n'
        f'Prompt: {prompt}\nCompletion: {completion}\nScore:',
        model
    )
    if resp:
        nums = re.findall(r'\d+', resp)
        if nums:
            return min(10, max(0, int(nums[0]))) / 10.0
    return 0.0

def generate_reference(prompt, model):
    ref = ask_ollama(
        prompt, model,
        "Complete this sentence in 3-8 words. Be natural and grammatical. Reply ONLY with the completion."
    )
    if ref and len(ref.split()) <= 15:
        return ref.strip()
    return None


# ══════════════════════════════════════════════════════════════════════
# Tuning Engine
# ══════════════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    "the cat sat on the",
    "the king and the queen",
    "once upon a time there was a",
    "the boy walked to the",
    "she opened the door and saw a",
    "in the garden there was a",
    "the old man looked at the",
    "they went to the market to",
    "the little girl picked up the",
    "he ran as fast as he could to",
    "the dog barked at the",
    "she smiled and said",
]

PARAMS = ['attn_threshold', 'route_band', 'content_band', 'trans_weight',
          'anti_repeat', 'prompt_boost', 'n_heads']
RANGES = {
    'attn_threshold': (0.15, 0.45),
    'route_band': (0.10, 0.30),
    'content_band': (0.40, 0.70),
    'trans_weight': (0.3, 1.5),
    'anti_repeat': (0.2, 1.0),
    'prompt_boost': (0.1, 0.6),
    'n_heads': (4, 12),
}


def random_config():
    return {k: lo + random.random() * (hi - lo) for k, (lo, hi) in RANGES.items()}

def apply_config(model, config):
    for k in PARAMS:
        if k in config:
            val = config[k]
            if k == 'n_heads':
                val = int(val)
            setattr(model, k, val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--model', default='devstral-small-2:24b')
    parser.add_argument('--eval-prompts', type=int, default=4)
    parser.add_argument('--ref-cycle', type=int, default=10,
                       help='Ask for reference completions every N iterations')
    args = parser.parse_args()
    
    print("█" * 60)
    print("  RIGOROUS RLM TUNING — Embedded Model + LLM Judge")
    print(f"  {args.iterations} iterations, ollama model: {args.model}")
    print("█" * 60)
    
    # Build RLM
    print("\n  Building RLM from corpus...")
    rlm = FastRLM()
    rlm.init_from_corpus(max_vocab=500)
    
    # Run tuning
    best_score = 0.0
    best_config = {k: getattr(rlm, k) for k in PARAMS}
    history = []
    refs_collected = 0
    bigrams_added = 0
    
    print(f"\n  Tuning {len(PARAMS)} parameters over {args.iterations} iterations...\n")
    
    for it in range(args.iterations):
        # Every ref_cycle iterations, ask Ollama for reference completions
        if it > 0 and it % args.ref_cycle == 0:
            for prompt in random.sample(TEST_PROMPTS, 2):
                ref = generate_reference(prompt, args.model)
                if ref:
                    n = rlm.inject_reference(ref)
                    bigrams_added += n
                    refs_collected += 1
                    print(f"    [ref] injected {n} bigrams from \"{ref[:50]}...\"")
        
        # Random configuration or perturb best
        if it == 0:
            config = {k: getattr(rlm, k) for k in PARAMS}
        elif random.random() < 0.3:
            config = random_config()
        else:
            config = dict(best_config)
            for k in PARAMS:
                lo, hi = RANGES[k]
                config[k] += (random.random() - 0.5) * (hi - lo) * 0.5  # wider search
                config[k] = max(lo, min(hi, config[k]))
        
        apply_config(rlm, config)
        
        # Evaluate on a few prompts
        scores = []
        eval_prompts = random.sample(TEST_PROMPTS, args.eval_prompts)
        
        for prompt in eval_prompts:
            result, _ = rlm.generate(prompt, max_tokens=8)
            s = score_coherence(prompt, result, args.model)
            scores.append(s)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        history.append(avg_score)
        
        if avg_score > best_score:
            best_score = avg_score
            best_config = dict(config)
            print(f"  [{it+1:>4d}] avg={avg_score:.3f} ★ best | "
                  f"thr={config['attn_threshold']:.2f} "
                  f"n_h={int(config['n_heads'])} "
                  f"tw={config['trans_weight']:.2f} "
                  f"ar={config['anti_repeat']:.2f}")
        elif it % 5 == 0:
            print(f"  [{it+1:>4d}] avg={avg_score:.3f}    | "
                  f"thr={config['attn_threshold']:.2f} "
                  f"n_h={int(config['n_heads'])} "
                  f"tw={config['trans_weight']:.2f}")
    
    # Final results
    apply_config(rlm, best_config)
    
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Iterations: {args.iterations}")
    print(f"  References collected: {refs_collected}")
    print(f"  Bigrams injected: {bigrams_added}")
    print(f"  Best score: {best_score:.3f}")
    print(f"\n  Best configuration:")
    for k in PARAMS:
        v = best_config[k]
        if k == 'n_heads':
            print(f"    {k}: {int(v)}")
        else:
            print(f"    {k}: {v:.3f}")
    
    # Show best outputs
    print(f"\n  Sample generations with best config:")
    for prompt in TEST_PROMPTS[:4]:
        result, _ = rlm.generate(prompt, max_tokens=10)
        print(f"    \"{prompt}\" → \"{result}\"")
    
    # Score trend
    if len(history) > 1:
        trend = [sum(history[:i+1])/(i+1) for i in range(len(history))]
        print(f"\n  Score trend: {trend[0]:.3f} → {trend[-1]:.3f} "
              f"({'improving' if trend[-1] > trend[0] else 'flat'})")


if __name__ == "__main__":
    main()
