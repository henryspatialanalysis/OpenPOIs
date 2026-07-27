# Conflation match scoring

How a candidate (OSM, Overture) pair gets a score, how the threshold was
calibrated, and what to check when changing any of it. The type component has
its own deep-dive in [type-affinity-metric.md](type-affinity-metric.md).

Reworked 2026-07-26/27. Everything below reflects the `20260729` run.

## Pipeline

1. **Candidate generation** — BallTree radius search over centroids, gated by
   the per-label radius from `match_radii.csv`, capped at
   `conflation.max_radius_m` (200 m). Purely spatial: identifiers play no part,
   so a pair further apart than the radius is never considered however strong
   the other evidence.
2. **Scoring** — a weighted sum of four components, each in [0, 1].
3. **Selection** — greedy one-to-one above `conflation.min_match_score`.

## The four components

| component | source | range |
|---|---|---|
| distance | `1 - d/radius`, clipped | [0, 1] |
| name | max of 4 `token_set_ratio` comparisons (name×name, brand×brand, both crosses); **0.5 neutral when all are null** | [0, 1] |
| type | derived affinity table, see [type-affinity-metric.md](type-affinity-metric.md) | [0, 1] |
| identifier | 1.0 if website, phone or Wikidata brand id agrees; else 0.0 | {0, 1} |

## Per-pair weights

Weights are selected per candidate pair, not applied globally:

| | distance | name | type | identifier |
|---|--:|--:|--:|--:|
| both sides carry an identifier | 0.1667 | 0.3333 | 0.3333 | 0.1667 |
| otherwise (`conflation.*_weight`) | 0.20 | 0.40 | 0.40 | 0.00 |

Both sets sum to exactly 1.0, so the composite is a convex combination and an
all-agreeing pair scores exactly 1.0 (6,829 pairs did in `20260728`).

The split exists because identifier evidence is decisive when present but
missing on roughly five of six pairs — OSM carries a website on 17% of rows and
a phone on 14%, against 83% / 92% on the Overture side. A single fixed
identifier weight would spend score mass on a blank for most candidates.

> **Do not fold a neutral constant into a component that is usually absent.**
> Until 2026-07-26 `compute_identifier_scores` was a stub returning 0.5 for
> every pair, with a stale docstring claiming Overture had no website/phone
> fields. At `identifier_weight: 0.20` that added a flat 0.10 to *every*
> composite — a fifth of the score carrying no information, and a large part of
> why scores compressed into 0.5-0.9 with nothing above 0.9.

## Identifier comparison

A pair is *comparable* if any one kind is present on both sides; the subscore is
1.0 if any comparable kind agrees. Agreement on one kind wins even if another
disagrees — a branch phone differing from a head-office number is not
counter-evidence of the same weight as a matching website.

Normalisation (`normalize_website` / `normalize_phone` / `normalize_wikidata`):

- **website** — lowercase; drop scheme and leading `www.`; **cut query string and
  fragment**; strip trailing slashes. The query string matters: Overture ships
  `westernunion.com/?utm_source=bingmaps&utm_medium=pml-yex` on 34,162 rows,
  which could never equal OSM's plain `westernunion.com`.
- **phone** — digits only, last 10 (NANP); discarded if shorter.
- **wikidata** — uppercase, trimmed, must match `^Q[1-9][0-9]*$`; anything else
  is discarded rather than compared.

### Identifiers are not unique — do not short-circuit on them

The tempting rule "same website within 200 m ⇒ auto-match" is wrong. Only 72% of
Overture websites belong to a single POI; 633,941 POIs (6.2%) share one with
100+ others:

```
34,162  westernunion.com/?utm_source=bingmaps...
10,834  subway.com/en-us
 9,739  7-eleven.com
 8,639  usps.com
```

For a chain the website is a *brand* identifier, not a location one. Two
different Subways 150 m apart would both satisfy such a rule. The same argument
applies to `brand:wikidata`, which is a chain id by construction. Name and type
must still agree, and greedy one-to-one selection stops both binding to the same
record.

An **inverse-frequency weighting** would be the principled improvement — an
identifier held by one POI is near-conclusive, one held by 34,162 is worth
almost nothing. That is the `u` probability of Fellegi-Sunter; not yet built.

## The 0.70 threshold, and how it was calibrated

`conflation.min_match_score` was raised 0.50 → 0.70 on 2026-07-26 after manually
reviewing 30 sampled matches per score band on the `20260728` run:

| band | n | precision | character |
|---|--:|--:|---|
| 0.9-1.0 | 1,051,507 | ~100% | near-exact name pairs |
| 0.8-0.9 | 386,837 | ~97% | same brand, trivial wording differences |
| 0.7-0.8 | 345,121 | ~87% | mixed but mostly sound |
| 0.6-0.7 | 235,782 | ~60% | *First United Methodist Church* ↔ *BAPS Shri Swaminarayan Mandir*, 19 m apart |
| 0.5-0.6 | 191,651 | ~33% | *Sprint* ↔ *All American Smile*; *Aldo* ↔ *Joe Fresh* |

Raising the floor dropped 420,360 matches. The POIs are not lost — they return
as single-source rows, so the total row count *rises*.

**Re-run this sampling whenever the weights, the affinity table or the threshold
change.** A deterministic sample (`ORDER BY hash(unified_id) LIMIT 30` inside the
band filter) makes it repeatable. Note `USING SAMPLE n ROWS` in DuckDB may apply
before the `WHERE`, returning one row per band — order by a hash instead.

## Unnamed OSM POIs cap at 0.80

A null name scores the neutral 0.5, so with the no-identifier weights a perfect
distance and perfect type still tops out at
`0.20 + 0.40 + 0.40x0.5 = 0.80`. Only 1.1% of unnamed-OSM matches reach 0.80+,
against 73.3% of named ones.

This is **not** evidence that those matches are bad. Sampling 30 of them in the
0.7-0.8 band found ~83% clearly correct — unnamed `amenity=fuel` 3 m from an
Overture "Sunoco" with a matching label, and similar for churches, playgrounds,
banks and schools. For an unnamed feature, type plus tight distance genuinely is
the evidence. 89,235 of the 119,348 unnamed matches in that band are within 25 m.

So do **not** add a rule requiring a named OSM POI — it would discard ~119k
mostly-correct matches. The cleaner fix is to renormalise the weights over the
components that actually have evidence, exactly as the identifier component now
does, letting these score on merit instead of being capped. Not yet built.

## The shadow matcher shares this configuration

`apply_change_detection.py` reads the same `conflation.distance_weight` /
`name_weight` / `type_weight` / `identifier_weight` keys and calls the same
`compute_match_scores`. Retuning the main matcher therefore moves ghost matching
too: the 2026-07 weight change took shadow matches from 57,760 to 62,248 (+7.8%)
with the mean penalty factor essentially unchanged (0.1720 → 0.1731).

Its threshold is separate (`conflation.change_detection.min_shadow_match_score`,
raised to 0.70 alongside the main one) and its scores are **not** on quite the
same scale: it passes all-zero L0 bits and no affinity table, so its type score
is binary on exact `shared_label` equality, and it supplies no identifier arrays
so it always uses the no-identifier weight set.

## Where this should go next

Fellegi-Sunter probabilistic record linkage would dissolve the weighting
question rather than answering it: estimate m = P(agree | match) and
u = P(agree | non-match) per field, score as `sum log(m/u)`. Weights fall out of
the data, the output is calibrated log-odds so the threshold has a probabilistic
meaning, distance enters as a binned agreement pattern rather than a linear
term, and the inverse-frequency identifier weighting comes for free. It is a
rewrite of the scoring core and wants a clean type-affinity input first.
References are in [type-affinity-metric.md](type-affinity-metric.md).
