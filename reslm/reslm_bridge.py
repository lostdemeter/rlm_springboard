"""
Resonant Language Model — Full φ→Transformer Bridge

Architecture:
  Layer 0: φ-exponent spectrum at coarse scale → semantic axis layout
  Layer 1: φ-exponent spectrum at fine scale → detail refinement

  Token embeddings:   γ·token_id mod 2π → H-element complex vector
  Position encoding:  γ·log(pos+1) mod 2π → phase rotation
  Attention:          content+position resonance, signed, no softmax
  FFN:                φ-spectrum → 4-state {+1,+0,-0,-1} activations
  Output:             integer dot product, top-k with tie-breaking

Zero training. Zero randomness. Identical output on every machine.

Run:  python3 reslm_bridge.py ["prompt"]
"""

import math
import sys
from collections import Counter

TWOPI = 2.0 * math.pi
PHI = (1 + 5 ** 0.5) / 2

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073]

BLOCK = 4  # dimensions per semantic axis (simplified from 6)

# ── Primitives ─────────────────────────────────────────────────────────

def phase(key, gamma):
    return (gamma * key) % TWOPI

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, TWOPI - d) / math.pi

def hash_key(text):
    h = 0
    for c in text:
        h = (h * 31 + ord(c)) & 0x7FFFFFFF
    return h

# ── φ-Exponent Spectrum ────────────────────────────────────────────────

def phi_spectrum(x, n_zeros=8):
    """a_e(x) for φ-exponents e ∈ [-24, ..., 24] at scale x."""
    exps = range(-24, 25, 2)
    spec = {e: 0.0 for e in exps}
    for n in range(min(n_zeros, len(GAMMAS))):
        g = GAMMAS[n]
        denom = 0.25 + g * g
        p = g * math.log(x)
        contrib = -2 * math.sqrt(x) * (0.5 * math.cos(p) + g * math.sin(p)) / denom
        for e in exps:
            coh = (math.cos(p - math.pi * e / 12) + 1) / 2
            spec[e] += contrib * coh
    return spec

# ── Semantic Axes from φ-Spectrum ─────────────────────────────────────

def spectrum_to_axes(spec, layer_idx=0, n_axes=12):
    """Convert φ-spectrum to semantic axis weights.
    
    Layer 0 (coarse): use top n_axes by |a_e|, mapping SIGN+MAG to 4-state.
    Layer 1 (fine): use remaining axes for detail.
    
    Returns: list of (axis_name, 4-state_weight_vector)
    """
    exps = sorted(spec.keys(), key=lambda e: -abs(spec[e]))
    
    if layer_idx == 0:
        active = exps[:n_axes]
    else:
        start = n_axes * layer_idx
        active = exps[start:start + n_axes]
    
    max_abs = max(abs(spec[e]) for e in active) if active else 1.0
    axes = []
    
    for i, e in enumerate(active):
        val = spec[e]
        sign = 1.0 if val >= 0 else -1.0
        frac = i / max(1, len(active))
        
        # 4-state quantization: first third = +1, middle = 0, last = -1
        if frac < 0.33:
            mag_state = 1.0
        elif frac < 0.66:
            mag_state = 0.0
        else:
            mag_state = -1.0
        
        # BLOCK-dimensional axis: [SIGN, MAG, φ⁴_flag, φ²_flag]
        axis = [
            sign,
            sign * mag_state,
            1.0 if abs(e) == 4 else 0.0,  # outer channel flag
            1.0 if abs(e) == 2 else 0.0,  # inner channel flag
        ]
        axes.append((f'φ^{e:+d}', axis))
    
    return axes

# ── Resonant Embedding ─────────────────────────────────────────────────

class ResonantEmbedding:
    """Token embeddings: γ·token_id → H heads → 2H real values."""
    def __init__(self, vocab_size, n_heads=8):
        self.V = vocab_size
        self.H = min(n_heads, len(GAMMAS))
        self.gammas = GAMMAS[:self.H]
        self._cache = {}
    
    def __call__(self, token_id):
        if token_id not in self._cache:
            vec = []
            for g in self.gammas:
                p = phase(token_id + 1, g)
                vec.extend([math.cos(p), math.sin(p)])
            self._cache[token_id] = vec
        return list(self._cache[token_id])
    
    @property
    def dim(self):
        return 2 * self.H


# ── Resonant Attention (Content + Position) ───────────────────────────

class ResonantAttention:
    """Content-aware signed attention via phase resonance.
    
    q_phase(h, i, tok) = γ_h · (token_id_i + pos_i + 1) mod 2π
    k_phase(h, j, tok) = γ_h · (token_id_j + pos_j + 1) mod 2π
    
    If circ_dist < threshold, positions i and j resonate.
    Weight = 1 - dist/threshold (linear decay).
    """
    def __init__(self, n_heads=8, threshold=0.25):
        self.H = min(n_heads, len(GAMMAS))
        self.gammas = GAMMAS[:self.H]
        self.threshold = threshold
    
    def forward(self, x, token_ids, pos_ids):
        """x: T × D list of lists. Returns attended T × D."""
        T = len(x)
        if T == 0:
            return []
        D = len(x[0])
        Dh = D // self.H
        out = [list(row) for row in x]
        
        for h in range(self.H):
            g = self.gammas[h]
            q_phases = [phase(token_ids[i] + pos_ids[i] + 1, g) for i in range(T)]
            start = h * Dh
            
            for i in range(T):
                total_w = 0.0
                wsum = [0.0] * Dh
                
                for j in range(i + 1):  # causal
                    d = circ_dist(q_phases[i], q_phases[j])
                    if d < self.threshold:
                        w = 1.0 - d / self.threshold
                        total_w += w
                        for di in range(Dh):
                            if start + di < D:
                                wsum[di] += w * x[j][start + di]
                
                if total_w > 0:
                    w_factor = 1.0 / total_w
                    for di in range(Dh):
                        if start + di < D:
                            out[i][start + di] = wsum[di] * w_factor
        
        return out


# ── φ-Weighted Feed-Forward ───────────────────────────────────────────

class PhiFFN:
    """FFN with weights derived from φ-exponent spectrum.
    
    At each layer's scale x, the φ-spectrum determines which semantic
    axes are active. Each axis's SIGN+MAG activates specific dimensions
    of the hidden state.
    """
    def __init__(self, d_model, n_heads=8, layer_idx=0):
        self.D = d_model
        self.H = n_heads
        self.layer = layer_idx
        self.gammas = GAMMAS[:n_heads]
        
        # Compute φ-spectrum at layers' characteristic scales
        x = math.exp(2.0 + layer_idx * 1.5)
        self.spec = phi_spectrum(x, n_zeros=n_heads)
        self.axes = spectrum_to_axes(self.spec, layer_idx, n_axes=12)
    
    def forward(self, x):
        T = len(x)
        if T == 0:
            return []
        D = len(x[0])
        out = [list(row) for row in x]
        
        # Use φ-axes to gate activations
        for i in range(T):
            agg_phase = sum(out[i][:4]) % TWOPI
            
            for d in range(D):
                activation = 0.0
                for name, axis in self.axes:
                    sign, mag, outer, inner = axis
                    p = phase(d + 1, GAMMAS[min(hash_key(name) % self.H, self.H - 1)])
                    coh = 0.5 + 0.5 * math.cos(agg_phase - p)
                    activation += sign * abs(mag) * coh / len(self.axes)
                
                # Quantize to 4-state
                if activation > 0.5:
                    out[i][d] = 1.0
                elif activation > 0.0:
                    out[i][d] = 0.5
                elif activation > -0.5:
                    out[i][d] = -0.5
                else:
                    out[i][d] = -1.0
        
        return out


# ── Resonant Language Model ────────────────────────────────────────────

class ResLM:
    def __init__(self, vocab, n_heads=8, n_layers=2):
        self.vocab = list(vocab)
        self.t2i = {t: i for i, t in enumerate(vocab)}
        self.i2t = {i: t for i, t in enumerate(vocab)}
        self.V = len(vocab)
        self.H = min(n_heads, len(GAMMAS))
        self.D = 2 * self.H
        
        self.embed = ResonantEmbedding(self.V, self.H)
        self.attn = ResonantAttention(self.H, threshold=0.28)
        self.ffns = [PhiFFN(self.D, self.H, li) for li in range(n_layers)]
        self.gammas = GAMMAS[:self.H]
    
    def tokenize(self, text):
        return [self.t2i.get(c, 0) for c in text]
    
    def detokenize(self, ids):
        return ''.join(self.i2t.get(i, self.vocab[0]) for i in ids)
    
    def forward(self, token_ids):
        T = len(token_ids)
        if T == 0:
            return {i: 0 for i in range(self.V)}
        
        pos_ids = list(range(T))
        
        # Embed tokens
        x = [self.embed(tid) for tid in token_ids]
        
        # Layer 0: coarse attention + φ-FFN
        x = self.attn.forward(x, token_ids, pos_ids)
        x = self.ffns[0].forward(x)
        
        # Layer 1: fine attention + φ-FFN
        if len(self.ffns) > 1:
            x = self.attn.forward(x, token_ids, pos_ids)
            x = self.ffns[1].forward(x)
        
        # Output: integer dot product
        final = x[-1]
        scores = {}
        for tid in range(self.V):
            emb = self.embed(tid)
            s = 0.0
            for d in range(min(len(final), len(emb))):
                s += int(round(final[d])) * emb[d]
            scores[tid] = int(round(s))
        
        return scores
    
    def generate(self, prompt, max_tokens=40, top_k=5):
        tokens = self.tokenize(prompt)
        result = list(prompt)
        
        for step in range(max_tokens):
            scores = self.forward(tokens)
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            top = sorted_scores[:top_k]
            
            best_score = top[0][1]
            best_tokens = [t for t, s in top if s == best_score]
            next_token = min(best_tokens)
            
            # Stop conditions
            if self.i2t.get(next_token, '') in '\n':
                if len(tokens) > len(self.tokenize(prompt)) + 3:
                    break
            
            tokens.append(next_token)
            result.append(self.i2t[next_token])
        
        return ''.join(result)


# ── Demo ───────────────────────────────────────────────────────────────

def demo():
    vocab = list(" abcdefghijklmnopqrstuvwxyz.,!?-'\n")
    model = ResLM(vocab, n_heads=8, n_layers=2)
    
    print("█" * 60)
    print("  RESONANT LANGUAGE MODEL — FULL φ BRIDGE")
    print("  φ-spectrum → semantic axes → signed attention")
    print(f"  Vocab={model.V}, H={model.H}, D={model.D}, layers={len(model.ffns)}")
    print("█" * 60)
    
    prompt = sys.argv[1] if len(sys.argv) > 1 else "the "
    print(f"\n  Prompt: \"{prompt}\"")
    
    result = model.generate(prompt, max_tokens=50, top_k=3)
    continuation = result[len(prompt):]
    print(f"  Output: \"{continuation}\"")
    print(f"  FULL:   \"{result}\"")
    
    # Determinism
    result2 = model.generate(prompt, max_tokens=50, top_k=3)
    print(f"\n  {'✓ Deterministic' if result == result2 else '✗ NON-DETERMINISTIC!'}")
    
    # φ-spectrum analysis
    print(f"\n{'='*60}")
    print(f"  φ-Spectrum Semantic Axes (Layer 0, x=e²)")
    spec0 = phi_spectrum(math.exp(2), n_zeros=8)
    axes0 = spectrum_to_axes(spec0, 0, 6)
    for name, axis in axes0:
        states = ['+1' if v == 1 else '-1' if v == -1 else ' 0' for v in axis[:2]]
        flags = []
        if axis[2]: flags.append('φ⁴')
        if axis[3]: flags.append('φ²')
        f = f"[{','.join(flags)}]" if flags else ''
        print(f"    {name:>6s}: [{', '.join(states)}]{f}")
    
    print(f"\n  Layer 1 (x=e³·⁵):")
    spec1 = phi_spectrum(math.exp(3.5), n_zeros=8)
    axes1 = spectrum_to_axes(spec1, 1, 6)
    for name, axis in axes1[:6]:
        states = ['+1' if v == 1 else '-1' if v == -1 else ' 0' for v in axis[:2]]
        print(f"    {name:>6s}: [{', '.join(states)}]")
    
    # Frequency analysis of output
    if result:
        char_counts = Counter(result)
        print(f"\n  Output character distribution:")
        for c, n in char_counts.most_common(8):
            print(f"    '{c}': {n}")
    
    print(f"\n{'='*60}")
    print(f"  Properties")
    print(f"{'='*60}")
    print(f"""
  Embedding:      γ·token_id mod 2π → complex vector
  Position:        γ·log(pos+1) → phase rotation
  Attention:       content+position phase resonance (no softmax)
  FFN weights:     φ-exponent spectrum → 4-state axes
  Semantic axes:   {len(axes0)} per layer (from (±4,±2) identity)
  Layers:          {len(model.ffns)} (coarse → fine φ-scale)
  Training:        NONE — all weights from φ + ζ structure
  Randomness:      NONE — deterministic on every machine
  Parameters:      0 trainable parameters
""")


if __name__ == "__main__":
    demo()
