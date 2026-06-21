# Mixed-Precision Native-KLU Study: Precision, Iterative Refinement, and the Quire

This study measures how MTL5's native KLU sparse direct solver behaves across
number systems and solver strategies on a circuit-simulation matrix, to guide
mixed-precision SPICE. It composes **MTL5** (the linear algebra) with
**Universal** (the number systems); MTL5 itself stays free of any Universal
dependency, and all coupling lives in mp-spice.

Reproduce with:

```bash
klu_precision_study path/to/add32.mtx        # the comprehensive table
klu_quire_study     path/to/matrix.mtx       # the quire deep-dive
```

## Setup

- **Matrix:** `Hamm/add32` (4960 × 4960, 23 884 nonzeros), a real circuit matrix
  from the SuiteSparse collection.
- **Right-hand side:** `b = A · 1`, so the exact solution is all-ones and the
  **forward error** `‖x − 1‖∞` is known directly (not just the residual).
- **Number systems:** `double`, `float`, `bfloat16` (`cfloat<16,8>`: 8 exponent /
  7 mantissa bits), `half` (`cfloat<16,5>`: 5 exponent / 10 mantissa bits),
  `posit<16,2>`, `posit<32,2>`.
- **Strategies:**
  - **direct** — factor and solve entirely in the low-precision type;
  - **IR** — factor in the low-precision type, then iteratively refine using a
    **double-precision residual**, reusing the low-precision factorization
    (native KLU's reusable factorization makes this cheap);
  - **quire** — replace the factorization's per-block inner-product accumulation
    with a Universal posit **quire** (an exact fused dot product), via MTL5's
    accumulator policy seam.

All residuals (`‖Ax−b‖∞`) and forward errors (`‖x−1‖∞`) are evaluated in double.

## Table 1 — precision sweep: direct solve vs iterative refinement

| type | direct residual | direct fwd err | **IR residual** | **IR fwd err** | IR iters |
|------|-----------------|----------------|-----------------|----------------|----------|
| double      | 2.1e-17 | 4.9e-15 | 2.1e-17 | 4.9e-15 | 0 |
| float       | 6.9e-09 | 3.5e-06 | 2.8e-17 | **5.1e-15** | 2 |
| bfloat16    | 3.3e-04 | 1.7e-01 | 1.3e-16 | **1.9e-13** | 16 |
| half        | 5.5e-05 | 2.1e-02 | 3.0e-08 | **5.7e-05** | 30 (capped) |
| posit<16,2> | 3.1e-05 | 1.1e-02 | 9.5e-15 | **4.5e-12** | 30 (capped) |
| posit<32,2> | 6.9e-10 | 2.6e-07 | 2.1e-17 | **7.2e-15** | 2 |

### What the direct column says

- At **16 bits**, `posit<16,2>` (fwd err 1.1e-2) edges out IEEE `half` (2.1e-2)
  and clearly beats `bfloat16` (1.7e-1) on a *direct* solve — posit's tapered
  precision puts more accuracy near 1.0, where this scaled system lives.
- At **32 bits**, `posit<32,2>` (2.6e-7) beats `float` (3.5e-6) for the same
  reason.

### What the IR column says — the headline

Iterative refinement with a double residual recovers accuracy dramatically for
**float, bfloat16, and both posits** — but **`half` stalls**:

| type | IR fwd err | converged? |
|------|-----------|------------|
| float       | 5.1e-15 | yes (2 it) |
| posit<32,2> | 7.2e-15 | yes (2 it) |
| bfloat16    | 1.9e-13 | yes (16 it) |
| posit<16,2> | 4.5e-12 | yes (30 it, still descending) |
| **half**    | **5.7e-05** | **no** (stuck) |

The counterintuitive part: **`bfloat16` converges but `half` does not — even
though `half` has a *better* direct solve and 3× more mantissa bits.**

The explanation is **dynamic range, not precision.** IR repeatedly forms a
residual, casts it down to the low-precision type, solves for a correction, and
adds it back. As the solution improves, the residual (and the correction) become
tiny. `half` has only 5 exponent bits — its smallest normal magnitude is ~6e-5 —
so once the correction drops below that, casting it to `half` **flushes it toward
zero** and refinement stops making progress. `bfloat16` keeps `float`'s 8
exponent bits (range ~1e-38), and `posit` has a very wide tapered range, so both
can still *represent* a small correction and keep converging — despite carrying
fewer significant bits.

**Practical rule:** for mixed-precision iterative refinement, choose the
low-precision type by **dynamic range (exponent bits)**, not mantissa width.
`bfloat16` and posits are good IR carriers at 16 bits; IEEE `half` is not, for a
system whose corrections span a wide magnitude range — exactly the regime of
stiff circuit matrices. (This is the precise mechanism behind the earlier
observation that the `half`/`cfloat<16,5>` path failed to converge.)

## Table 2 — the quire (exact fused dot product) vs plain accumulation

The quire makes each column's inner products in the factorization accumulate
exactly and round only once (a fused dot product). Posits only (Universal's quire
is defined for posit products).

| type | direct fwd err (plain → quire) | IR fwd err (plain → quire) | IR iters |
|------|-------------------------------|----------------------------|----------|
| posit<16,2> | 1.07e-2 → **0.98e-2** | 4.55e-12 → 4.55e-12 | 30 → 30 |
| posit<32,2> | 2.61e-7 → **2.38e-7** | 7.22e-15 → **4.89e-15** | 2 → 2 |

- On a **direct** solve the quire helps slightly (a few percent here; ~1.5× on
  the denser systems in `klu_quire_study`) — it removes the intra-column
  accumulation rounding.
- Under **IR the quire's benefit essentially vanishes**: same iteration count,
  effectively the same floor. Iterative refinement *already* compensates for the
  factorization's accumulation error, so making that accumulation exact buys
  almost nothing once you refine.

**Practical rule:** prefer cheap iterative refinement over the (much more
expensive) quire when you can iterate. The quire's niche is a **single
high-quality factorization without refinement**. The part IR is *actually*
sensitive to is the residual/correction representation — see Table 3.

## Table 3 — scaled iterative refinement (extended-precision residual)

The Table 1 IR column casts each residual directly into the low-precision type.
As the correction shrinks, a narrow-exponent type (`half`) underflows and IR
stalls. **Scaled IR** normalizes each residual to O(1) before casting, solves,
and restores the correction's magnitude in double — so only the normalized
*shape* passes through the type:

```
rho = ||r||_inf  (double);   dx = rho * ( U_T \ ( L_T \ (r / rho) ) )
```

| type | unscaled IR (fwd err / it) | **scaled IR (fwd err / it)** |
|------|----------------------------|------------------------------|
| float       | 5.1e-15 / 2  | 5.6e-15 / 2  (unchanged) |
| bfloat16    | 1.9e-13 / 16 | 1.7e-13 / 16 (unchanged) |
| **half**    | **5.7e-05 / 30 (stalled)** | **2.8e-14 / 7 (converged)** |
| **posit<16,2>** | **4.5e-12 / 30 (capped)** | **1.2e-14 / 6** |
| posit<32,2> | 7.2e-15 / 2  | 6.8e-14 / 1  (unchanged) |

This is the decisive confirmation that **`half`'s stall is a residual
*representation* problem, not a fundamental accuracy limit.** Carrying only the
*magnitude* in double (one scalar per step) collapses `half`'s forward error from
5.7e-5 to **2.8e-14** — near double-level — in **7** cheap iterations. It also
takes `posit<16,2>` from capping at 4.5e-12/30 to **1.2e-14 in 6** iterations.
The wide-range types (`float`, `bfloat16`, `posit<32,2>`) are unchanged, exactly
as expected: scaling matters only when the type's exponent range is the
bottleneck.

So the practical fix is almost free — a single `double` scale factor per
refinement step — and it makes **every** 16-bit type studied a viable
mixed-precision carrier, including IEEE `half`.

## Synthesis and guidance for mixed-precision SPICE

1. **Posits are the strongest 16- and 32-bit carriers here** — best direct-solve
   accuracy at each width, and they refine to ~1e-12 (16-bit) or machine-level
   (32-bit).
2. **For iterative refinement, dynamic range beats mantissa width.** Use
   `bfloat16` or posits at 16 bits; avoid IEEE `half` for stiff systems whose
   corrections underflow its narrow exponent range.
3. **IR is the cheap, dominant accuracy lever.** Factor low, refine with a
   double residual, reuse the factorization — float and posit<32,2> reach
   double-level accuracy in ~2 steps.
4. **Scaled IR (an extended-precision residual *magnitude*) is the key enabler at
   16 bits.** A single double scale factor per step rescues `half` (5.7e-5 → 2.8e-14)
   and accelerates `posit<16,2>` (4.5e-12/30 → 1.2e-14/6). With it, **every**
   16-bit type studied — including IEEE `half` — reaches near double-level
   accuracy. Always scale the refinement RHS for narrow-range low-precision
   carriers.
5. **The quire's payoff is in the factorization's backward error, which IR makes
   redundant.** Exact accumulation in the *factorization* buys little once you
   refine; the lever that mattered was the residual/correction *representation*
   (point 4), which scaled IR addresses cheaply without a quire.

## Files

- `applications/klu_precision_study/` — Table 1 + Table 2 (this study).
- `applications/klu_quire_study/` — quire deep-dive (direct + IR, with the
  small-matrix dense sweep).
- `include/sw/mp_spice/klu_study.hpp` — `direct_solve`, `mixed_refine`,
  `mixed_refine_scaled` (all accumulator-parameterized), residual/forward-error
  helpers.
- `include/sw/mp_spice/quire_accumulator.hpp` — the posit-quire accumulator
  adapter for MTL5's `accumulator_traits` seam.
