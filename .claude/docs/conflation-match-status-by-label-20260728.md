# Conflation match-status by shared_label (2026-07-28)

Fourth iteration: **two non-POI exclusions, no scoring changes**
(`versions.conflation = 20260730`). Supersedes
[the 2026-07-27 table](conflation-match-status-by-label-20260727.md).

- **Source**: `~/data/openpois/conflation/20260730/conflated.parquet` (14,613,331 rows).
- **Input**: OSM rated snapshot 4,764,221 — 5,015,126 less the residential-landuse
  exclusion (250,905) — plus Overture `20260722_tax3` (12,606,804). Same model fit
  (`20260727_by_shared_label`) and same match weights/cutoff as `20260729`.

## What changed

Two independent exclusions, applied together in one re-conflation:

- **A — residential landuse.** Unnamed POIs of private-prone types whose
  representative point falls inside a `landuse=residential` polygon. 250,905 rows
  (5.00%) off the rated snapshot. See the "Exclusion" section of
  [data-sources.md](data-sources.md).
- **B — wildcard to inclusion sets.** The `amenity`/`office`/`leisure`/`tourism`
  `*` crosswalk rows were replaced with explicit rows on a destination-vs-object
  criterion; only `shop` and `healthcare` keep a catch-all. 61,764 rows lost their
  `shared_label` and were removed by `drop_unlabeled`. See
  [taxonomy-setup.md](taxonomy-setup.md).

## Run comparison

| run | matched | OSM-only | Overture-only | total | match % |
|---|--:|--:|--:|--:|--:|
| 20260727 (taxonomy overhaul) | 2,332,474 | 2,445,604 | 9,556,257 | 14,334,335 | 16.3% |
| 20260728 (+ new scoring, cutoff 0.50) | 2,210,898 | 2,567,180 | 9,719,796 | 14,497,874 | 15.2% |
| 20260729 (+ cutoff 0.70) | 1,790,538 | 2,987,540 | 10,144,456 | 14,922,534 | 12.0% |
| **20260730 (+ exclusions A and B)** | **1,787,072** | **2,678,337** | **10,147,922** | **14,613,331** | **12.2%** |

The ledger reconciles exactly:

```
Change A (rated snapshot)          -250,905
Change B (drop_unlabeled)           -61,764
                         subtotal  -312,669
less matched -> overture-only        +3,466
                         expected  -309,203
                         observed  -309,203
```

The 3,466 is the interesting term: when an excluded OSM POI had been *matched*, the
row is not deleted — it demotes to Overture-only. Hence Overture-only gained
precisely what matched lost, and the matched segment moved only −0.19%. The
exclusions removed noise without disturbing the matched core, which is what you
would expect if they are hitting unnamed private features that rarely matched
anything.

`drop_unlabeled` removed 298,812 rows total (237,048 already unlabeled before this
change, plus B's 61,764).

## Labels that moved

Losses are the two exclusions landing:

| shared_label | delta | cause |
|---|--:|---|
| Swimming Pool | −126,187 (−47.1%) | A |
| Recreation | −73,560 (−8.0%) | A (`pitch`, `track`) |
| Other Amenity | −59,877 (−28.4%) | B (street furniture) + A (`fountain`) |
| Playground | −30,811 (−18.5%) | A |
| Other Professional | −30,106 (−7.6%) | B (`office=yes`) |
| Park | −11,619 (−2.4%) | A (`garden`) |

Gains are B's 165 new explicit rows pulling values *out* of the catch-alls into
specific labels — the half of the change that adds precision rather than removing
rows: Government Office +3,383 (`polling_station`, `public_building`), Public Safety
+3,157 (`prison`, `mountain_rescue`), Arts Venue +2,878 (`studio`), Other Healthcare
+1,937 (`nursing_home`), Hotel +1,184 (`leisure=resort`), Counseling +571
(`office=therapist`), Social Club +672, Arcade +308.

## Top 30 by total

| shared_label | matched | OSM-only | Overture-only | total | match % | delta |
|---|--:|--:|--:|--:|--:|--:|
| Restaurant | 188,189 | 37,887 | 658,920 | 884,996 | 21.3% | +34 |
| Recreation | 34,686 | 640,530 | 174,483 | 849,699 | 4.1% | -73,560 |
| Home Service | 6,709 | 10,027 | 797,866 | 814,602 | 0.8% | +1,088 |
| Place of Worship | 198,443 | 119,800 | 335,473 | 653,716 | 30.4% | +277 |
| Hair and Beauty | 58,379 | 18,232 | 509,705 | 586,316 | 10.0% | +247 |
| Other Financial | 22,233 | 15,458 | 477,653 | 515,344 | 4.3% | +464 |
| Park | 75,864 | 302,524 | 101,608 | 479,996 | 15.8% | -11,619 |
| Car Repair | 51,916 | 13,207 | 409,439 | 474,562 | 10.9% | +389 |
| Specialty Store | 25,325 | 18,947 | 428,380 | 472,652 | 5.4% | +104 |
| Real Estate | 7,559 | 7,409 | 390,609 | 405,577 | 1.9% | +429 |
| Other Healthcare | 5,304 | 5,864 | 389,308 | 400,476 | 1.3% | +1,937 |
| School | 80,679 | 73,069 | 220,031 | 373,779 | 21.6% | +1,077 |
| Other Professional | 6,439 | 35,933 | 322,895 | 365,267 | 1.8% | -30,106 |
| Community Center | 11,976 | 26,366 | 278,626 | 316,968 | 3.8% | -3 |
| Fast Food | 137,394 | 27,505 | 109,670 | 274,569 | 50.0% | +0 |
| Clothing Store | 35,025 | 19,210 | 215,467 | 269,702 | 13.0% | +2 |
| Hotel | 55,817 | 25,740 | 173,966 | 255,523 | 21.8% | +1,184 |
| Historic Site | 4,747 | 66,185 | 178,100 | 249,032 | 1.9% | +77 |
| Other Shop | 57,811 | 79,610 | 89,371 | 226,792 | 25.5% | +16 |
| Fitness Center | 18,298 | 7,129 | 172,925 | 198,352 | 9.2% | +228 |
| Car Dealer | 24,252 | 6,351 | 160,152 | 190,755 | 12.7% | -2 |
| Gas Station | 59,087 | 48,892 | 82,750 | 190,729 | 31.0% | -2 |
| Cemetery | 965 | 183,057 | 1,174 | 185,196 | 0.5% | +6 |
| Hardware | 6,672 | 4,241 | 173,414 | 184,327 | 3.6% | +0 |
| Legal Service | 6,821 | 6,570 | 167,234 | 180,625 | 3.8% | +319 |
| Bar | 32,211 | 11,537 | 136,170 | 179,918 | 17.9% | +88 |
| Cafe | 37,442 | 13,028 | 114,226 | 164,696 | 22.7% | +83 |
| Dentist | 19,887 | 5,111 | 137,046 | 162,044 | 12.3% | -8 |
| Convenience Store | 35,241 | 52,113 | 74,299 | 161,653 | 21.8% | +0 |
| Other Amenity | 2,695 | 96,180 | 52,348 | 151,223 | 1.8% | -59,877 |
| **TOTAL** | **1,787,072** | **2,678,337** | **10,147,922** | **14,613,331** | **12.2%** | **-309,203** |

## Notes

- **Single-source labels: 1** (Car Rental), unchanged. Overture still has no
  car-rental category.
- **A third change rode along in this run.** `min_shadow_match_score` was raised to
  0.70 after `20260729` and takes effect on the next `make conflate`, which was this
  one: shadow matches 62,248 → 30,340, mean penalty factor 0.1731 → 0.1962. It moves
  Overture *confidence values*, not row counts, so the ledger above is unaffected —
  but a `20260729`→`20260730` diff of confidences confounds all three changes. See
  [TODO.md](../TODO.md).
- **Ghosts were regenerated** under the new crosswalk: same 787,743 rows (detection
  is crosswalk-independent), but labelled ghosts fell 609,389 → 596,255, so ~13k
  fewer are eligible for shadow matching. `20260729`'s exact ghost input is preserved
  as `ghost_osm/20260724/ghosts_prewildcard.parquet`.
- **False-exclusion evidence for A** is in [data-sources.md](data-sources.md): of the
  22 eligible POIs in the `openpois-validator` pilot, 6 were removed and all 6 were
  `unverifiable`; neither eligible `exists` POI was touched. Small denominator —
  directional only.
- **`conf_mean` cannot validate A.** Dropped rows skew high-confidence, because the
  turnover model scores name-change stability and unnamed POIs have no name to
  change.
- Nothing was published from this run: no `conflated_partitioned/`, no pmtiles,
  `versions.source_coop` unchanged. `20260729` is retained intact — the
  `openpois-validator` pilot round is pinned to it.
