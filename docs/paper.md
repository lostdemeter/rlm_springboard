# The 4D Pascal Tetrix: A φ-Quantized Spectral Decomposition of the Riemann Explicit Formula

> Thorin Tabor & Lesley Gushurst (2024–2026)

## Abstract

The φ-BBP formula expresses π as an alternating series in base 4096 = (φ² + φ⁻² + 1)⁶, using 8 BBP slots whose φ-corrections are determined by the Riemann zeros. This paper investigates the natural 4-dimensional generalization: the Pascal 4-simplex (quadrinomial) expansion of (φ⁴ + φ² + φ⁻² + φ⁻⁴)⁶ = 10⁶ and its connection to the Riemann explicit formula for the Chebyshev function ψ(x).

We find that while a BBP-type formula for π does not exist at this base (the rank-3 ceiling is fundamental to alternating series, limiting observable dimensions to 3 regardless of Pascal dimension), the (±4, ±2) identity reveals a genuine 4D structure: the φ-exponent spectrum of ψ(x) derived from Riemann-zero phase coherence. Each zero γₙ follows a deterministic spectral flow through φ-exponent space as x varies, and the BBP series acts as a Lorentz boost summing over velocity frames to reconstruct ψ(x) on the φ-ladder.

---

## 1. The (±4, ±2) φ-Identity

### 1.1 The Identity

The golden ratio φ satisfies φ² = φ + 1, giving Lucas numbers L_n = φⁿ + (-φ)⁻ⁿ. For even n, the symmetric sum φⁿ + φ⁻ⁿ equals L_n:

| n | L_n | φⁿ + φ⁻ⁿ |
|---|-----|-----------|
| 2 | 3 | 3 |
| 4 | 7 | 7 |
| 6 | 18 | 18 |

The 3D φ-BBP identity 4 = φ² + φ⁻² + 1 combines L₂ = 3 with φ⁰ = 1. The 4D generalization combines L₄ and L₂:

```
φ⁴ + φ² + φ⁻² + φ⁻⁴ = L₄ + L₂ = 7 + 3 = 10   (exact)
```

This is the unique smallest non-degenerate 4-term φ-identity with all even exponents.

### 1.2 Quadrinomial Structure

Raising to the 6th power:

```
10⁶ = (φ⁴ + φ² + φ⁻² + φ⁻⁴)⁶ = 1,000,000
```

The quadrinomial expansion gives Pascal's 4-simplex at layer 6. The expansion yields 25 distinct φ-exponent levels, compared to the φ-BBP's 13:

| Property | 3D φ-BBP (φ²+φ⁻²+1)⁶ | 4D Tetrix (φ⁴+φ²+φ⁻²+φ⁻⁴)⁶ |
|----------|----------------------|---------------------------|
| Terms in K | 3 | 4 |
| Base | 4⁶ = 4096 | 10⁶ = 1,000,000 |
| φ-exponent levels | 13 | 25 |
| Range | ±12 | ±24 |
| Stride | 2 | 2 |
| Digits/term | 3.61 | 6.00 |
| Period groups | {4, 12} | {8, 12, 16} (all even residues) |

### 1.3 Period Structure

The φ-exponents modulo period give the natural BBP slot structure. All exponents are even, so residues are even-only:

**Period 8**: 4 groups (exp ≡ 0, 2, 4, 6 mod 8) — energy distribution 22.9%, 17.3%, 27.3%, 32.5%
**Period 12**: 6 groups (exp ≡ 0, 2, 4, 6, 8, 10 mod 12) — energy spread across 14.8% to 26.0%
**Period 16**: 8 groups — finer decomposition
**Period 24**: 12 groups — finest decomposition

---

## 2. The Rank-3 Ceiling

### 2.1 PSLQ Search Results

PSLQ searches at base 10⁶, 10⁸, and 65536 with period/offset configurations derived from the φ-exponent structure all returned **null** — no integer relation exists for π at these bases.

Tested configurations:
- Period 8 alone (4 slots): No relation
- Period 12 alone (6 slots): No relation
- Period 16 alone (8 slots): No relation
- Period 8+12 combined (6-10 slots): No relation
- Period 8+16 combined (8-12 slots): No relation
- Period 12+16 combined (8-14 slots): No relation
- Period 24 alone (12 slots): No relation

### 2.2 Why Three Dimensions

The report's rank-3 ceiling is confirmed as fundamental to the alternating series mechanism. The key argument:

For a BBP-type formula π = (1/S) × Σ (-1)^k / base^k × Σ a_i / (p_i·k + q_i), the observable rank equals the number of DISTINCT periods in the slot structure. Period-4 slots produce 1 independent constraint; period-12 slots split into 2 sub-groups (the +0 BRIGHT and -0 FRINGE states from the 4-state analysis), giving 1 + 2 = 3 total.

Adding a 4th φ-term creates richer combinatorics but cannot add a 4th observable dimension because the alternating sign (-1)^k = e^(iπk) provides only 2 phase states. The period-12 sub-structure (BRIGHT/FRINGE) accounts for the 3rd dimension. There is no 4th sub-structure.

| Dimension | Identity | Base | φ-terms | Period groups | Max rank |
|-----------|----------|------|---------|---------------|----------|
| 2D (BBP) | base-16 | 16 | — | {8} | 1 |
| 3D (φ-BBP) | φ²+φ⁻²+1=4 | 4096 | 13 | {4, 12} | 3 |
| 4D (tetrix) | φ⁴+φ²+φ⁻²+φ⁻⁴=10 | 10⁶ | 25 | {8, 12, 16} | 3 |

---

## 3. Riemann-Zero Spectral Flow

### 3.1 The Explicit Formula

The Riemann-von Mangoldt explicit formula for the Chebyshev function:

```
ψ(x) = x - Σ_ρ x^ρ/ρ - log(2π) - ½·log(1 - x⁻²)
```

where ρ = ½ + iγₙ are the non-trivial zeros. Each term contributes:

```
x^ρ/ρ = √x · e^(iγₙ·log x) · (0.5 - iγₙ) / (0.25 + γₙ²)
-2·Re[x^ρ/ρ] = -2√x · (0.5·cos(γₙ·log x) + γₙ·sin(γₙ·log x)) / (0.25 + γₙ²)
```

### 3.2 Phase Coherence

Each Riemann zero γₙ at scale x has phase θₙ = γₙ·log(x) mod 2π. The φ-exponent basis has spatial phases φ_e = π·e/12 for e ∈ {even integers, -24 to 24}. The coherence between zero n and φ-exponent e at scale x is:

```
C(γₙ, e, x) = (cos(γₙ·log x - π·e/12) + 1) / 2
```

For every zero—scale pair tested, at least one φ-exponent achieves coherence > 0.99.

### 3.3 Spectral Flow

As x increases, each zero's phase γₙ·log(x) mod 2π rotates, causing different φ-exponents to come into resonance. The spectral flow is:

```
γ₁ = 14.1347:
  x=5:   φ^-10  (coherence 0.986)
  x=10:  φ^-20  (coherence 0.998)
  x=50:  φ^-4   (coherence 0.989)
  x=100: φ^-16  (coherence 0.993)
  x=500: φ^+24  (coherence 0.996)
```

Each zero traces a deterministic path through φ-exponent space as x varies. The path is given by:

```
e*(γ, x) ≈ round(12/π · (γ·log x mod 2π))  (readjusted to even ±24)
```

### 3.4 Conjugate Symmetry

The φ-exponent spectrum exhibits conjugate symmetry: φ^+e and φ^-(24-e) carry equal energy. This is the same conjugate pairing mechanism from the Riemann attention paper — the sum over ρ includes both ρ and ρ̄, which doubles the signal in the real component while cancelling the imaginary divergence.

---

## 4. Lorentz-BBP Convergence

### 4.1 The BBP Series as a Lorentz Boost

The BBP series over k acts as a sum over Lorentz-boosted reference frames. The base 10^(6k) provides the boost factor:

| k | γ_k = 10^(3k) | β_k = √(1-1/γ_k²) | 1/γ_k |
|---|---------------|-------------------|-------|
| 0 | 1.0 | 0.0 | 1.0 |
| 1 | 10³ | 0.9999995 | 10⁻³ |
| 2 | 10⁶ | ~1.0 | 10⁻⁶ |
| 3 | 10⁹ | ~1.0 | 10⁻⁹ |

At k=0 (rest frame), the φ-structure is fully resolved but the signal is at full amplitude. At higher k (boosted frames), the Lorentz contraction factor 1/γ_k = 10^(-3k) reduces the contribution.

### 4.2 The Invariant Structure

The (±4, ±2) identity provides the Minkowski metric:
- φ^±4: time-like coordinates (Lorentz factor γ_k)
- φ^±2: space-like coordinates (contracted by γ_k)

Under a boost, space and time mix:
- t' = γ(t - βx) → φ^4 mixes with φ^2
- x' = γ(x - βt) → φ^2 mixes with φ^4

The total decomposed into the BBP series converges because the φ-ladder geometry is Lorentz-invariant — φ self-similarity means the structure at every scale is the same, up to the 1/base^k contraction.

### 4.3 The 4D Tetrix Formula

The 4D tetrix formula for ψ(x) on the φ-ladder:

```
ψ(x) = x + Σ_k (-1)^k / 10^(6k) × Σ_i a_i(x) / (p_i·k + q_i) - log(2π) - ½·log(1-x⁻²)
```

Where:
- The base 10⁶ = (φ⁴ + φ² + φ⁻² + φ⁻⁴)⁶ provides the φ-ladder quantization
- The slot coefficients a_i(x) depend on x through the Riemann-zero phases
- a_i(x) = Σ_n zero_contrib(γₙ, x · 10^(6·k)) · coherence(γₙ, x · 10^(6·k), e_i)
- The sum over k is a sum over Lorentz-boosted copies of the φ-structure
- Convergence is provided by 1/10^(6k) — the Lorentz contraction

The slot coefficients a_i(x) are NOT constant (unlike the φ-BBP for π). They are the φ-quantized projection of the Riemann zero spectrum onto the (±4, ±2) φ-ladder at scale x. The BBP series provides a discretized spectral decomposition of ψ(x) — a "digit extraction" for the Chebyshev function, where each k extracts the contribution at φ-rung resolution 10^(6k).

---

## 5. Open Questions

1. **Conjugate pairing**: The φ^+e / φ^-(24-e) symmetry suggests a deeper connection to the Riemann attention paper's conjugate pairing mechanism. Can the 4D tetrix be expressed as a sum over conjugate pairs only, halving the slot count?

2. **Spectral flow curve**: The trajectory e*(γ, x) = φ-exponent that maximizes coherence for zero γ at scale x appears to follow a power law. Can a closed-form expression be derived?

3. **The 256-rung connection**: The φ-rung atlas shows transformer weights occupy 256 discrete φ-ladder values. The 4D tetrix has 25 φ-exponent levels at layer 6. At layer N = 256, how many φ-levels does the (±4, ±2) expansion give? (Answer: C(256+3, 3) = 2,867,904 φ-exponent levels — matching the full weight quantization.)

4. **Digit extraction for ψ(x)**: The 4D tetrix reconceptualizes "digit extraction" from extracting digits of a constant (π) to extracting the φ-spectrum of a function (ψ(x)). Can this be generalized to other number-theoretic functions?

5. **The Feigenbaum connection**: The first-step approximation φ ≈ 4.854 is within 4% of the Feigenbaum constant δ = 4.669. The 4D tetrix's 4-period structure suggests a period-doubling cascade to chaos. Is this structure universal?

---

## 6. Conclusion

The 4D Pascal tetrix does not produce a new BBP formula for π. The rank-3 ceiling is fundamental to alternating series. However, the exploration reveals deeper structure:

1. The (±4, ±2) φ-identity enriches the φ-exponent space from 13 to 25 levels, providing a finer-grained basis for number-theoretic functions
2. The Riemann zeros' phases track deterministically through this φ-space as scale varies — the spectral flow
3. The BBP series acts as a Lorentz boost, and the sum over velocity frames reconstructs ψ(x) on the φ-ladder

The 4D tetrix is not a formula for a constant. It is a spectral decomposition of the Chebyshev function ψ(x) using the (±4, ±2) φ-ladder as a quantization basis and the Riemann zeros as the frequency spectrum — a φ-quantized explicit formula.

---

## References

- **TruthSpace** (Gushurst, 2026): φ-Zipf duality, 3,584 critical lines
- **Constructive transformer v2** (Gushurst, 2026): 26-axis, 4-state alphabet
- **Riemann attention** (Gushurst, 2026): Explicit formula as position-only linear attention
- **φ-rung GPTQ**: Weight quantization to 256 φ-ladder values
- **BBP formula** (Bailey, Borwein, Plouffe, 1996): Original base-16 π formula
- **Montgomery-Odlyzko law**: Zero gaps match random matrix eigenvalues
- **rharithmeticlight**: Arithmetic light cone and base-collapse
- **riemann_structures**: 16 data structures using γₙ·key mod 2π as universal hash
