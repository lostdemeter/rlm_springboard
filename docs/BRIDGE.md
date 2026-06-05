# φ-Spectrum → Constructive Transformer Bridge

## The Idea

The constructive transformer (constructive_transformerv2) uses hand-placed
4-state integer weights for attention, with no training and no softmax.
The φ-exponent spectrum a_e(x) from the 4D tetrix provides a principled
source for placing those weights — each Riemann zero γₙ determines a
spectral profile that maps directly to the 4-state axis layout.

## Mapping

| φ-spectrum a_e(x) | Constructive transformer |
|-------------------|------------------------|
| φ-exponent e | Semantic axis (NUMBER, LEXCLASS, TENSE, ...) |
| sign(a_e) | SIGN value (+1 or -1) |
| |a_e|/max|a| | MAGNITUDE (strength of the axis) |
| |a_e| > threshold | Active axis (carries signal) |
| |a_e| < threshold | "Hole" axis (zeroed, negligible) |
| Outer φ^±4 | φ^4 flag — routing gate (EXPAND/CONTRACT) |
| Inner φ^±2 | φ^2 flag — agreement marker |
| γₙ · log(x) mod 2π | Riemann phase — determines which axes resonate |

## 4-State Axis Layout (BLOCK=6)

Each axis occupies 6 dimensions:

```
idx  role           values
──────────────────────────────
0    SIGN          +1 or -1       (from sign of a_e)
1    MAGNITUDE     ±|a_e|/max    (from spectral energy)  
2    ANYSTATE      1.0            (always active)
3    φ^4 FLAG      2.0 if |e|=4  (outer channel, strong routing)
4    φ^2 FLAG      2.0 if |e|=2  (inner channel, agreement)
5    reserved      0.0
```

The (±4,±2) φ-identity means φ^±4 and φ^±2 have special roles —
they're the STRONG routing channels (outer) and AGREEMENT channels (inner),
mirroring the Lorentz boost structure where φ^±4 is time-like and φ^±2 is
space-like.

## Attention Routing

Each Riemann zero γₙ drives one attention head. The head determines
which axes resonate by computing:

```
phase(i) = γₙ · hash(axis_name_i) mod 2π
alignment(i,j) = cos(phase(i) - phase(j))
```

If alignment > 0.5, axis i routes its SIGN+MAG to axis j.
This is O(N·H) — each head independently checks which axes resonate.

## LM Head

The LM head is a signed integer dot product between the attended vector
and each output candidate. No softmax:

```
score(candidate) = Σ_i vec_attended[i] · vec_candidate[i]
```

Margins are guaranteed to be integers because all values are
from {+1, +0, -0, -1} with BLOCK=6 structure.

## Current Status

- ✅ φ-exponent spectrum → axis mapping works
- ✅ 4-state vector generation from a_e(x) works
- ✅ Signed attention with Riemann phase alignment works
- ⚠ LM head produces equal scores (needs feature-specific routing)
- ❌ Full language demo (needs integration with specific axis conventions)

The remaining step: integrate the φ-spectrum axis layout with the
constructive transformer's BLOCK=6 conventions for NUMBER, LEXCLASS,
and TENSE axes, so that the attention routing and LM head produce
meaningful margins.
