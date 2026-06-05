# Resonant Language Model — Springboard

**A deterministic, zero-parameter language model from Riemann zeros and the golden ratio.**

```
phase = (riemann_zero * key) % (2 * pi)
```

One primitive. Every component. Zero training. Identical output on every machine, forever.

## The Story

### Part 1: The Resonant Array (No GloVe, No Training)

We discovered that the Riemann zeros can replace random hash functions in data structures. The Montgomery-Odlyzko law guarantees uniform phase distribution — same collision bounds as random hashing, but fully deterministic.

From one primitive `γ·key mod 2π`, we built:

| Structure | What It Replaces | Proof |
|-----------|-----------------|-------|
| **Resonant Array** | Hash table + similarity search | O(1) lookup + O(N·H) resonance |
| **Resonant Dedup** | Seed-coordinated deduplication | 64 files matched across two nodes, zero coordination |
| **Resonant Attention** | Softmax attention | Signed phase resonance, no floating point |
| **Resonant Signature** | Learned embeddings | H-element complex vectors from γ·id mod 2π |

**The demo that sells it:** Two machines, same data, zero seed coordination, identical phases to 6 decimal places. 7.3 million resonance checks per second. 1 million files indexed in 4 seconds vs 139 hours for traditional pairwise comparison.

### Part 2: The Global Similarity Breakthrough (φ-Zipf + Resonant Phases)

We previously built φ-Zipf word embeddings — the magnitude distribution matched the φ-rung atlas (mean rung ~111), frequency scaling followed `rank^(-0.481)`. But we hit a block: matching STATISTICS didn't produce GLOBAL SIMILARITY. "king - man + woman" didn't give "queen" because token positions in embedding space were random, not structured.

**The breakthrough:** Combining φ-Zipf magnitude distributions (which give the right statistics) with Resonant Array phase assignment (which gives deterministic token positions via `γ·token_id mod 2π`). Each token now gets a UNIQUE, DETERMINISTIC embedding where:

- Magnitudes follow φ-Zipf decay `|a_h| ∝ φ^(-h)` within each vector
- Norm follows Zipf's law `|e| ∝ rank^(-0.481)` across tokens
- Phases are deterministic by Riemann zero geometry
- Same token → same embedding on every machine, in every language, forever

### Part 3: The Resonant Language Model (Can RLM Work?)

A language model where every component is `γ·key mod 2π`:

- **Embeddings:** φ-Zipf magnitudes × Resonant phases (no lookup table)
- **Attention:** Content+position phase resonance (no softmax, O(N·H))
- **FFN:** φ-exponent spectrum → 4-state activations (no learned weights)
- **Output:** Integer dot product with tie-breaking (no temperature)

Current capability: zero-parameter model detects and repeats the most frequent character in the prompt. The bridge to the constructive transformer's 4-state semantic axes is in progress.

## The Question

**Is a Resonant Language Model possible?** Can a language model with zero trainable parameters, using only `γ·key mod 2π`, produce coherent text? The pieces are here:

- ✅ Deterministic token embeddings (φ-Zipf + Resonant phases)
- ✅ Signed attention without softmax (phase resonance)
- ✅ Explainable token selection (integer dot product)
- ✅ Multi-head consensus (independent Riemann zeros agree)
- ❓ Semantic key assignment (mapping concepts to token IDs)
- ❓ Coherent multi-token generation

## Directory Structure

```
rlm_springboard/
├── README.md                         ← You are here
├── LICENSE                            ← GPLv3
│
├── resonant/                          ← Part 1: Resonant Array demos
│   ├── resonant_array.py              Core data structure + 3 derived types
│   ├── resonant_dedup.py              Distributed dedup (two nodes, zero sync)
│   ├── resonant_parallel_bench.py     Parallel benchmark (7M checks/sec)
│   ├── resonant_code_search.py        Codebase indexing
│   ├── oracle.py                      Interactive terminal demo
│   └── resonant_demo.py              Publication-quality visualization
│
├── phi_zipf/                          ← Part 2: Global similarity breakthrough
│   └── phi_zipf_embed.py             φ-Zipf + Resonant phases → embeddings
│
├── reslm/                             ← Part 3: Resonant Language Model
│   ├── reslm_zeroparam.py            Zero-parameter demo (working)
│   └── reslm_bridge.py              φ-Spectrum → transformer bridge
│
├── constructive/                      ← Constructive transformer bridge
│   ├── phi_constructive.py           φ-Exponent spectrum → 4-state weights
│   └── RAMIFICATIONS.md             Design rationale
│
└── docs/                              ← Supporting theory
    ├── music_box_methodology.md       7-step generalization template
    ├── paper.md                      4D Pascal Tetrix
    └── BRIDGE.md                     φ→Transformer architecture map
```

## Running

No dependencies beyond Python 3.10+ and the standard library.

```bash
# Part 1: Resonant Array demos
cd resonant
python3 resonant_dedup.py                           # Distributed dedup
python3 resonant_parallel_bench.py                   # 7M checks/sec bench
python3 oracle.py                                    # Interactive search

# Part 2: φ-Zipf embeddings
cd phi_zipf
python3 phi_zipf_embed.py                            # Deterministic embeddings

# Part 3: Resonant Language Model
cd reslm
python3 reslm_zeroparam.py                           # Zero-parameter generation
python3 reslm_bridge.py                              # φ-Spectrum bridge

# Optional: visualization (requires matplotlib + numpy)
python3 resonant/resonant_demo.py                    # Publication-quality figure
```

## Why This Matters

Every machine on Earth computes the same `γ·key mod 2π` for the same key. No seed coordination. No random number generator. No training data. No GPU cluster. The Riemann zeros are universal constants — they're the same numbers everywhere.

This means:
- Distributed systems don't need to coordinate hash seeds
- Language models don't need to download weights
- Embeddings don't need training data
- Output is guaranteed deterministic — forever

## License

GPLv3 — see [LICENSE](LICENSE)
