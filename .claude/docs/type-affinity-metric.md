# Type-affinity metric for conflation matching

Internal note, 2026-07-26. Options for replacing the current three-valued type
score, why we picked one, and what the references are.

## The problem

`compute_type_scores` in [match.py](../../src/openpois/conflation/match.py) scores
a candidate pair on POI type with three values: exact `shared_label` equality → 1.0,
overlap in the `top_level_matches.csv` L0 bitmask → 0.5, otherwise 0.0.

After the 2026-07 taxonomy overhaul this is the weakest part of the matcher:

- **Tier 1 got better.** 70.9% of matched pairs now agree on label, because the
  crosswalk actually resolves Overture's fine categories instead of dumping them
  in catch-alls. That justifies raising the type weight.
- **Tier 2 got worse.** A finer taxonomy (93 → 102 labels) means two sources
  describing the same place land on adjacent-but-different labels more often. The
  largest disagreements are all legitimate near-misses — Fast Food ↔ Restaurant
  (52,326 pairs), Park ↔ Recreation (both directions, ~17k), Tire Store ↔ Car
  Repair, Gas Station ↔ Convenience Store — and each is penalised as if unrelated.
- **Tier 2 is also unsafe.** The bitmask is label-agnostic: it only knows "OSM key
  `leisure` touches Overture L0 `lifestyle_services`", so `leisure=park` against a
  hair salon collects the same 0.5 as a genuine fitness-centre match. 8,838 pairs
  score Park ↔ Hair and Beauty today.

A hand-assigned 1 / 0.7 / 0 would fix the symptom by assertion. The options below
derive the number instead.

## Option 1 — information-content similarity over the Overture hierarchy

Our `shared_label`s project onto the Overture category tree, so semantic distance
is computable. The classic measures split into path-based and
information-content-based.

**Path-based** (Wu & Palmer; Leacock & Chodorow) treat every tree edge as equally
significant. That fails here because our tree is badly unbalanced:
`food_and_drink/restaurant` holds ~900k POIs, `casual_eatery/bakery` ~62k. One
edge is not one unit of meaning.

**Information-content-based** (Resnik; Lin; Jiang & Conrath) weight by how
surprising the shared ancestor is. With IC(c) = −log P(c) estimated from our own
snapshot's category frequencies:

```
lin(a, b) = 2 · IC(LCS(a, b)) / (IC(a) + IC(b))
```

Resnik uses IC(LCS) alone, so every pair under a given parent scores identically
regardless of their own specificity. Lin normalises by both nodes and lands in
[0, 1], which is what we want for a score component. Budanitsky & Hirst's
comparison of these measures on lexical-relatedness tasks is the usual reference
for preferring the IC family.

A `shared_label` maps to a *set* of Overture paths (Car Repair draws from both
`shopping/vehicle_parts_store` and `travel_and_transportation/vehicle_service`),
so label-level similarity aggregates over path pairs. **Use max, not a
count-weighted mean** — see the validation note below; the mean dilutes identity
to 0.586.

Output is a precomputed label × label table: O(1) at match time, reproducible
from data we already hold, no hand-tuned constants.

## Option 2 — empirical affinity from identifier-confirmed pairs

Rather than a proxy for semantic distance, measure the quantity we actually care
about: *when two records are genuinely the same place, how often does OSM say A
and Overture say B?*

Pairs where a normalised website or phone matches exactly are ground truth that
is **independent of the type score**, so using them does not circularly reinforce
the matcher. 579,254 of 2,332,474 matched pairs (24.8%) qualify.

The natural first choice is pointwise mutual information (Church & Hanks) with
Bouma's normalisation. **We tried it and rejected it** — see "Why not PMI" below.
What works is a row-normalised confusion matrix: of the confirmed matches whose
OSM label is `a`, what fraction carry Overture label `b`, scaled so the modal
counterpart is 1.

Caveat: websites and phones skew to named chain businesses, so Playground,
Cemetery and Public Restroom have thin or empty cells.

### Why not PMI

nPMI measures association *relative to chance*, not similarity, and its
attainable maximum depends on the marginals. Two failures on real cells:

- **Identity is not 1.** nPMI(Restaurant, Restaurant) = 0.755, because OSM
  Restaurant also legitimately pairs with Overture Bar, Cafe and Fast Food. With
  n = 69,473 the empirical term dominates any blend, so exact label agreement
  would score **0.755 — lower than today's 1.0**. A straight blend regresses the
  single strongest type signal we have.
- **Anchoring on the diagonal explodes.** Dividing by nPMI(a,a) to force identity
  to 1 gives Other Shop ↔ Specialty Store = **2.746**, because catch-all labels
  have weak diagonals (Other Shop pairs with itself at only 0.213). Unbounded
  above, so unusable as a score component.

The row-normalised conditional is bounded [0, 1] by construction, needs no
anchoring, and preserves identity automatically wherever identity is genuinely
the modal counterpart.

### What the confusion matrix revealed

Row-normalising exposed crosswalk facts no hierarchy encodes:

| OSM label | modal Overture counterpart | n | self-pairing |
|---|---|--:|--:|
| Tire Store | **Car Repair** | 1,556 | Tire Store, 122 |
| Other Shop | **Specialty Store** | 9,490 | Other Shop, 442 |

When OSM says `shop=tyres` and the match is identifier-confirmed, Overture calls
it Car Repair 13× more often than Tire Store. The two projects draw the
tyre-shop/auto-service boundary in different places. That is a real, measurable
property of the sources, invisible to any tree-distance measure.

## Option 3 — Fellegi–Sunter record linkage

The textbook framing, which dissolves the weighting question rather than
answering it. For each comparison field estimate m = P(agree | true match) and
u = P(agree | non-match); the score becomes Σ log(m/u), a log-odds sum instead of
a weighted mean. Weights fall out of the data (typically by EM), the output is
calibrated so a threshold has probabilistic meaning, and distance enters as a
binned agreement pattern rather than a linear term. Winkler's string-comparator
extensions are the standard refinement; `fastLink` (Enamorado, Fifield & Imai)
and Splink (UK Ministry of Justice) are current implementations.

This would also fix a defect visible in the present scores: they span only
0.5–0.9, with nothing above 0.9 and 1.25M of 2.33M matches piled in the top
bucket, because the firing weights sum to 0.8 with `distance_weight: 0.0`. The
scale is not measuring confidence across its range.

## Decision

**Options 1 and 2 combined, with shrinkage; Option 3 as the follow-on.**

Full specification. Let `T` be the Overture snapshot POI total and `N(v)` the
number of POIs beneath tree node `v` (a path prefix of `(l0, l1, l2, l3)`):

```
IC(v)        = −ln( N(v) / T )                        information content
LCS(p, q)    = longest common prefix of paths p, q
lin(p, q)    = 2·IC(LCS(p,q)) / (IC(p) + IC(q))       Lin (1998), 0 if denom = 0

S_lin(A, B)  = max over p ∈ P(A), q ∈ P(B) of lin(p, q)
               where P(L) = Overture paths the crosswalk maps to label L
               S_lin = 0 if either label has no Overture paths

S_emp(a, b)  = n(a, b) / max_b′ n(a, b′)
               n(a, b) = identifier-confirmed matched pairs with OSM label a
               and Overture label b; row max over the same OSM label

S(a, b)      = max( S_lin(a,b),
                    ( n(a,b)·S_emp(a,b) + k·S_lin(a,b) ) / ( n(a,b) + k ) )
```

The outer `max` makes the empirical term **additive only** — it can lift a pair
but never push it below the hierarchy prior. Without it the metric is perverse:
`S_emp` measures *frequency* ("what does Overture usually call this?"), not
compatibility, so row-max normalisation punishes a rare but exact agreement.
Measured on the generated table before the floor was added:

| pair | blended | note |
|---|--:|---|
| Tire Store → Tire Store | 0.493 | two sources agreeing exactly … |
| Tire Store → Car Repair | 0.979 | … scored *below* two sources disagreeing |
| Other Shop → Other Shop | 0.223 | |
| Alternative Medicine → Alternative Medicine | 0.405 | |

With the floor, all 101 identity pairs sit at exactly 1.000 while the empirical
lifts survive unchanged (Tire Store → Car Repair 0.979, Other Shop → Specialty
Store 0.998, Gas Station → Convenience Store 0.065, where `S_lin` is 0).

and the matcher uses `type_score(i,j) = S(osm_label(i), overture_label(j))`,
replacing the 1.0 / 0.5 / 0.0 tiers. Unlabelled on either side → 0.

Notes on the shape:

- **`S_lin` is symmetric, `S_emp` and therefore `S` are not.** That is correct:
  the matcher always compares an OSM label against an Overture label, and the
  sources' disagreements are directional (OSM Tire Store → Overture Car Repair is
  common; the reverse is not).
- **Identity needs no special case.** Where a label is its own modal counterpart,
  `S_emp = 1` and `S_lin = 1`, so `S = 1` for any `k`. Where it is *not* — Tire
  Store, Other Shop — that is a finding, not an error.
- **`n(a,b) = 0` ⟹ `S = S_lin`**, so unobserved cells fall back cleanly to the
  hierarchy prior; both zero ⟹ 0.
- `S_emp ≥ 0` by construction, so no flooring decision is required.

The IC similarity is a dense prior covering every label pair; the empirical term
pulls it toward observed reality wherever there is enough data; `k` controls how
much evidence is needed to move off the prior. Validation (below) shows neither
term alone is sufficient, which is the argument for the hybrid.

### Shrinkage constant: k = 100 (decided 2026-07-26)

`k` is the number of identifier-confirmed pairs at which the empirical term and
the hierarchy prior carry equal weight. Effect at the values considered:

| n(a,b) | weight on empirical, k=50 | **k=100** | k=500 |
|--:|--:|--:|--:|
| 7 | 12.3% | **6.5%** | 1.4% |
| 18 | 26.5% | **15.3%** | 3.5% |
| 301 | 85.8% | **75.1%** | 37.6% |
| 1,556 | 96.9% | **94.0%** | 75.7% |
| 9,490 | 99.5% | **99.0%** | 95.0% |

k = 100 keeps well-observed cells essentially empirical — Tire Store ↔ Car
Repair (n = 1,556) stays at 0.979, so the finding that Overture calls OSM tyre
shops "Car Repair" 13× more often than "Tire Store" survives — while sparse
cells stay on the prior (Hardware ↔ Furniture Store, n = 7, sits at 0.750 rather
than collapsing to its meaningless 0.003 observation).

k = 500 was rejected because it damps exactly the cells where the empirical
evidence is strongest and most surprising: Tire Store ↔ Car Repair falls to
0.914 and Gas Station ↔ Convenience Store — where the hierarchy is flatly wrong
at 0.000 and 520 confirmed matches prove co-location — halves from 0.071 to
0.040. k = 50 was rejected as slightly too eager to trust 50-observation cells.

`k` only affects the generated table, never the conflation run, so it can be
re-swept and diffed cheaply.

### Calibration loop

The empirical term is estimated from a *previous* conflation's output, so run N
calibrates the table used by run N+1. This is not circular: the confirmed pairs
are selected by website/phone agreement, which the type score plays no part in.
Rebuild the table whenever the taxonomy changes materially.

Fellegi–Sunter is the right destination but is a rewrite of the scoring core, and
it wants a clean type-affinity input regardless. Sequenced after.

## Validation findings that shaped the design

Measured on the 2026-07-27 conflation. `Lin(w)` is count-weighted mean over path
pairs, `Lin(max)` the maximum, `nPMI` the empirical term, `n` the number of
identifier-confirmed pairs.

| OSM label | Overture label | Lin(w) | Lin(max) | nPMI | n |
|---|---|--:|--:|--:|--:|
| Restaurant | Restaurant | 0.586 | 1.000 | +0.755 | 69,473 |
| Fast Food | Restaurant | 0.404 | 0.598 | +0.200 | 30,286 |
| Fast Food | Dessert Shop | 0.578 | 0.690 | +0.223 | 4,721 |
| Park | Recreation | 0.460 | 0.820 | +0.379 | 301 |
| Tire Store | Car Repair | 0.476 | 0.648 | +0.558 | 1,556 |
| Clinic | Other Healthcare | 0.491 | 0.952 | +0.616 | 2,337 |
| Other Shop | Specialty Store | 0.397 | 0.818 | +0.584 | 9,490 |
| Hardware | Furniture Store | 0.598 | 0.802 | −0.119 | 7 |
| **Gas Station** | **Convenience Store** | **0.000** | **0.000** | **+0.218** | **520** |
| Park | Hair and Beauty | 0.000 | 0.000 | n/a | 0 |
| Restaurant | Dentist | 0.000 | 0.000 | n/a | 0 |
| Cemetery | Bank | 0.000 | 0.000 | n/a | 0 |

Three findings:

1. **Use `max`, not the weighted mean.** Identity scores 0.586 under the mean,
   because a label spans many paths and cross-path pairs within the same label
   drag it down. `max` gives identity 1.000 and separates the near-misses
   cleanly (Clinic ↔ Other Healthcare 0.952, Park ↔ Recreation 0.820).
2. **The hierarchy alone is not enough.** Gas Station ↔ Convenience Store scores
   0.000 — different L0s, so the LCS is the root — yet 520 identifier-confirmed
   true matches say otherwise. Co-located fuel-and-shop sites are one real place;
   no tree distance recovers that.
3. **The empirical term alone is not enough.** Hardware ↔ Furniture Store has 7
   confirmed pairs and a meaningless −0.119. The prior has to carry sparse cells.

The unrelated pairs behave: Park ↔ Hair and Beauty scores 0.000 with zero
confirmed matches, so the 8,838 pairs currently collecting 0.5 from the bitmask
lose it, as intended.

A negative nPMI on a semantically close pair (Cafe ↔ Restaurant, −0.182) is not a
failure. It says that now the taxonomy resolves cafés correctly, a true café match
lands on Cafe ↔ Cafe, so the cross-pairing is *rarer* than chance and should not
earn type credit.

## References

- Wu, Z. & Palmer, M. (1994). Verb semantics and lexical selection. *ACL-94*.
- Resnik, P. (1995). Using information content to evaluate semantic similarity in
  a taxonomy. *IJCAI-95*.
- Jiang, J. & Conrath, D. (1997). Semantic similarity based on corpus statistics
  and lexical taxonomy. *ROCLING X*.
- Lin, D. (1998). An information-theoretic definition of similarity. *ICML-98*.
- Leacock, C. & Chodorow, M. (1998). Combining local context and WordNet
  similarity for word sense identification. In *WordNet: An Electronic Lexical
  Database*, MIT Press.
- Budanitsky, A. & Hirst, G. (2006). Evaluating WordNet-based measures of lexical
  semantic relatedness. *Computational Linguistics* 32(1).
- Church, K. & Hanks, P. (1990). Word association norms, mutual information, and
  lexicography. *Computational Linguistics* 16(1).
- Bouma, G. (2009). Normalized (pointwise) mutual information in collocation
  extraction. *Proceedings of GSCL*.
- Fellegi, I. & Sunter, A. (1969). A theory for record linkage. *JASA* 64(328),
  1183–1210.
- Winkler, W. (1990). String comparator metrics and enhanced decision rules in the
  Fellegi–Sunter model of record linkage. *Proceedings of the Section on Survey
  Research Methods, ASA*.
- Enamorado, T., Fifield, B. & Imai, K. (2019). Using a probabilistic model to
  assist merging of large-scale administrative records. *American Political
  Science Review* 113(2). (`fastLink`)
- Splink, UK Ministry of Justice — https://moj-analytical-services.github.io/splink/
