# Conflation match-status by shared_label (2026-07-27)

Third iteration: **taxonomy overhaul + reworked match scoring + cutoff 0.70**
(`versions.conflation = 20260729`). Supersedes
[the 2026-07-26 table](conflation-match-status-by-label-20260726.md).

- **Source**: `~/data/openpois/conflation/20260729/conflated.parquet` (14,922,534 rows).
- **Input**: OSM rated snapshot 5,015,126 (full-data `random_effects` refit
  `20260727_by_shared_label`) + Overture `20260722_tax3` (12,606,804; same release
  and allowlist as tax2, re-pulled to add `brand_wikidata`).

## Run comparison

| run | matched | OSM-only | Overture-only | total | match % |
|---|--:|--:|--:|--:|--:|
| 20260724 (before overhaul) | 1,955,542 | 2,239,304 | 9,539,255 | 13,734,101 | 14.2% |
| 20260727 (taxonomy overhaul) | 2,332,474 | 2,445,604 | 9,556,257 | 14,334,335 | 16.3% |
| 20260728 (+ new scoring, cutoff 0.50) | 2,210,898 | 2,567,180 | 9,719,796 | 14,497,874 | 15.2% |
| **20260729 (+ cutoff 0.70)** | **1,790,538** | **2,987,540** | **10,144,456** | **14,922,534** | **12.0%** |

The falling match % is intended. Manual review of 30 sampled matches per score band
found precision around 33% in 0.5-0.6 and 60% in 0.6-0.7, against ~87% at 0.7-0.8 and
~97%+ above. Raising the cutoff removes ~420k matches, the large majority of them
wrong; the POIs are not lost, they return as single-source rows, so the total grows.

## Scoring changes in this run

- **Type score** is now a derived affinity in [0, 1] rather than exact/0.5/0 tiers —
  see [type-affinity-metric.md](type-affinity-metric.md).
- **Identifier score** was a stub returning a constant 0.5 for every pair (20% of the
  composite carrying no information). It now compares normalised website, phone and
  `brand:wikidata` / `brand.wikidata`, scoring 1.0 if any agrees.
- **Weights are per-pair**: 0.167/0.333/0.333/0.167 (distance/name/type/identifier)
  where both sides carry an identifier, else 0.20/0.40/0.40/0.00.
- **Score range** went from a degenerate 0.5-0.9 (nothing above 0.9) to 0.7-1.0,
  mean 0.897, with 1,065,178 matches in the top decile.

## Full table

| shared_label | matched | OSM only | Overture only | total | match % | Δ vs 20260727 |
|---|--:|--:|--:|--:|--:|--:|
| Recreation | 35,857 | 713,801 | 173,601 | 923,259 | 3.9% | +27,153 |
| Restaurant | 188,180 | 37,842 | 658,940 | 884,962 | 21.3% | +38,463 |
| Home Service | 6,646 | 9,964 | 796,904 | 813,514 | 0.8% | +17,949 |
| Place of Worship | 198,315 | 119,543 | 335,581 | 653,439 | 30.3% | +23,275 |
| Hair and Beauty | 58,234 | 18,144 | 509,691 | 586,069 | 9.9% | +32,193 |
| Other Financial | 22,090 | 15,212 | 477,578 | 514,880 | 4.3% | +14,248 |
| Park | 75,780 | 314,469 | 101,366 | 491,615 | 15.4% | +27,662 |
| Car Repair | 51,706 | 12,869 | 409,598 | 474,173 | 10.9% | +10,295 |
| Specialty Store | 25,324 | 18,942 | 428,282 | 472,548 | 5.4% | +10,403 |
| Real Estate | 7,552 | 7,257 | 390,339 | 405,148 | 1.9% | +7,511 |
| Other Healthcare | 5,103 | 4,072 | 389,364 | 398,539 | 1.3% | +5,906 |
| Other Professional | 10,252 | 63,702 | 321,419 | 395,373 | 2.6% | +10,925 |
| School | 80,288 | 72,480 | 219,934 | 372,702 | 21.5% | +14,177 |
| Community Center | 11,980 | 26,359 | 278,632 | 316,971 | 3.8% | +42,236 |
| Fast Food | 137,394 | 27,505 | 109,670 | 274,569 | 50.0% | +6,650 |
| Clothing Store | 35,025 | 19,208 | 215,467 | 269,700 | 13.0% | +10,991 |
| Swimming Pool | 4,985 | 254,032 | 8,959 | 267,976 | 1.9% | +2,262 |
| Hotel | 55,183 | 24,611 | 174,545 | 254,339 | 21.7% | +6,865 |
| Historic Site | 4,740 | 66,119 | 178,096 | 248,955 | 1.9% | +18,814 |
| Other Shop | 57,810 | 79,611 | 89,355 | 226,776 | 25.5% | +4,809 |
| Other Amenity | 5,839 | 153,526 | 51,735 | 211,100 | 2.8% | +6,899 |
| Fitness Center | 18,296 | 7,131 | 172,697 | 198,124 | 9.2% | +16,600 |
| Car Dealer | 24,253 | 6,350 | 160,154 | 190,757 | 12.7% | +4,215 |
| Gas Station | 59,087 | 48,892 | 82,752 | 190,731 | 31.0% | +12,398 |
| Cemetery | 966 | 183,050 | 1,174 | 185,190 | 0.5% | +165 |
| Hardware | 6,672 | 4,241 | 173,414 | 184,327 | 3.6% | +3,780 |
| Legal Service | 6,584 | 6,421 | 167,301 | 180,306 | 3.7% | +4,566 |
| Bar | 32,050 | 11,483 | 136,297 | 179,830 | 17.8% | +8,693 |
| Playground | 2,807 | 160,611 | 3,373 | 166,791 | 1.7% | +968 |
| Cafe | 37,427 | 12,946 | 114,240 | 164,613 | 22.7% | +8,218 |
| Dentist | 19,887 | 5,111 | 137,054 | 162,052 | 12.3% | +3,898 |
| Convenience Store | 35,241 | 52,113 | 74,299 | 161,653 | 21.8% | +11,449 |
| Bank | 48,619 | 11,946 | 84,751 | 145,316 | 33.5% | +3,618 |
| Clinic | 24,713 | 14,630 | 105,130 | 144,473 | 17.1% | +4,868 |
| Massage Therapy | 3,951 | 2,327 | 137,257 | 143,535 | 2.8% | +10,163 |
| Government Office | 18,629 | 21,829 | 93,300 | 133,758 | 13.9% | +13,023 |
| Supermarket | 31,980 | 7,302 | 88,363 | 127,645 | 25.1% | +6,600 |
| ATM | 1,814 | 13,018 | 100,784 | 115,616 | 1.6% | +7,833 |
| Pet Store | 8,956 | 2,426 | 99,599 | 110,981 | 8.1% | +4,470 |
| Mental Health | 567 | 853 | 101,335 | 102,755 | 0.6% | +2,589 |
| Self-Storage | 9,601 | 23,882 | 63,115 | 96,598 | 9.9% | +2,094 |
| Furniture Store | 7,643 | 3,981 | 82,701 | 94,325 | 8.1% | +2,911 |
| Performing Arts | 6,789 | 7,147 | 75,277 | 89,213 | 7.6% | +10,477 |
| Public Safety | 25,055 | 24,329 | 38,982 | 88,366 | 28.4% | +5,321 |
| Dessert Shop | 9,578 | 2,552 | 70,726 | 82,856 | 11.6% | +6,093 |
| Florist | 3,973 | 1,359 | 73,697 | 79,029 | 5.0% | +2,349 |
| Public Restroom | 738 | 76,726 | 276 | 77,740 | 0.9% | +35 |
| Pharmacy | 18,353 | 12,039 | 47,242 | 77,634 | 23.6% | +3,126 |
| Campground | 8,291 | 37,332 | 31,960 | 77,583 | 10.7% | +1,896 |
| Post Office | 18,142 | 9,117 | 47,314 | 74,573 | 24.3% | +4,704 |
| Cell Phone Store | 12,073 | 4,883 | 56,550 | 73,506 | 16.4% | +1,909 |
| Sports Outlet | 4,821 | 1,721 | 66,221 | 72,763 | 6.6% | +4,715 |
| Car Wash | 9,144 | 12,247 | 45,409 | 66,800 | 13.7% | +1,703 |
| Print and Copy Shop | 4,141 | 1,425 | 60,484 | 66,050 | 6.3% | +2,478 |
| Bakery | 7,737 | 3,346 | 54,730 | 65,813 | 11.8% | +3,871 |
| Discount Store | 21,276 | 4,011 | 38,809 | 64,096 | 33.2% | +1,925 |
| Eye Care | 7,298 | 2,122 | 54,111 | 63,531 | 11.5% | +2,072 |
| University | 5,171 | 3,435 | 53,442 | 62,048 | 8.3% | +4,334 |
| Physical Therapy | 2,968 | 2,491 | 48,495 | 53,954 | 5.5% | +1,422 |
| Jewelry Store | 6,060 | 3,146 | 43,571 | 52,777 | 11.5% | +1,798 |
| Alternative Medicine | 5,008 | 2,420 | 45,329 | 52,757 | 9.5% | +1,345 |
| Thrift Store | 4,615 | 2,485 | 44,752 | 51,852 | 8.9% | +1,800 |
| Liquor Store | 11,133 | 4,413 | 33,162 | 48,708 | 22.9% | +1,854 |
| Arts Venue | 3,824 | 3,289 | 41,169 | 48,282 | 7.9% | +6,431 |
| Hospital | 6,539 | 1,432 | 38,396 | 46,367 | 14.1% | +1,396 |
| Childcare | 3,571 | 3,058 | 39,083 | 45,712 | 7.8% | +1,807 |
| Veterinarian | 8,509 | 1,445 | 34,842 | 44,796 | 19.0% | +1,758 |
| Garden Store | 2,248 | 3,710 | 34,833 | 40,791 | 5.5% | +825 |
| Kindergarten | 3,254 | 3,634 | 33,537 | 40,425 | 8.0% | +1,691 |
| Tattoo and Piercing | 3,428 | 1,076 | 34,860 | 39,364 | 8.7% | +2,068 |
| Library | 14,833 | 4,512 | 16,754 | 36,099 | 41.1% | +1,219 |
| Shoe Store | 5,055 | 2,975 | 26,898 | 34,928 | 14.5% | +1,398 |
| Museum | 9,041 | 5,634 | 19,694 | 34,369 | 26.3% | +4,170 |
| Bookstore | 3,854 | 1,443 | 26,366 | 31,663 | 12.2% | +1,450 |
| Funeral Services | 4,934 | 798 | 24,851 | 30,583 | 16.1% | +1,337 |
| Golf Course | 2,816 | 13,553 | 13,440 | 29,809 | 9.4% | +2,186 |
| Laundromat | 5,369 | 4,263 | 19,985 | 29,617 | 18.1% | +731 |
| Shopping Center | 1,847 | 3,336 | 19,767 | 24,950 | 7.4% | +1,094 |
| Event Venue | 1,041 | 5,563 | 16,315 | 22,919 | 4.5% | +2,865 |
| Farmers Market | 609 | 341 | 20,936 | 21,886 | 2.8% | +1,372 |
| Dry Cleaning | 3,669 | 3,742 | 12,902 | 20,313 | 18.1% | +415 |
| Charging Station | 813 | 12,357 | 5,624 | 18,794 | 4.3% | +260 |
| Stadium | 2,953 | 3,811 | 11,471 | 18,235 | 16.2% | +5,781 |
| Tire Store | 5,968 | 1,441 | 9,387 | 16,796 | 35.5% | +358 |
| Wholesale Store | 902 | 1,743 | 13,625 | 16,270 | 5.5% | +438 |
| Cannabis Dispensary | 2,785 | 1,771 | 11,510 | 16,066 | 17.3% | +516 |
| Animal Shelter | 1,007 | 492 | 13,184 | 14,683 | 6.9% | +597 |
| Nightclub | 1,185 | 1,006 | 12,449 | 14,640 | 8.1% | +1,529 |
| Social Club | 2,677 | 2,336 | 8,349 | 13,362 | 20.0% | +1,268 |
| Bike shop | 3,247 | 1,211 | 7,970 | 12,428 | 26.1% | +327 |
| Movie Theater | 4,439 | 1,053 | 6,355 | 11,847 | 37.5% | +981 |
| Marina | 2,146 | 3,542 | 5,121 | 10,809 | 19.9% | +765 |
| Dog Park | 2,173 | 5,866 | 2,562 | 10,601 | 20.5% | +707 |
| Market | 754 | 1,152 | 8,011 | 9,917 | 7.6% | +454 |
| Arcade | 1,093 | 1,014 | 4,693 | 6,800 | 16.1% | +671 |
| Bowling Alley | 1,856 | 460 | 3,844 | 6,160 | 30.1% | +474 |
| Casino | 893 | 428 | 4,311 | 5,632 | 15.9% | +526 |
| Counseling | 656 | 930 | 3,884 | 5,470 | 12.0% | +80 |
| Car Rental | 7 | 4,315 | 0 | 4,322 | 0.2% | +0 |
| Speech Therapist | 58 | 103 | 3,649 | 3,810 | 1.5% | +80 |
| Occupational Therapy | 62 | 123 | 3,493 | 3,678 | 1.7% | +87 |
| Maternity Center | 33 | 25 | 1,651 | 1,709 | 1.9% | +55 |
| **TOTAL** | **1,790,538** | **2,987,540** | **10,144,456** | **14,922,534** | **12.0%** | **+588,199** |

## Notes

- **Single-source labels: 1** (Car Rental), unchanged from the previous run.
  Overture has no car-rental category; its `matched` count fell from 1,000 to 7,
  which is the threshold correctly discarding matches that only ever paired OSM car
  rentals with auto dealers, restaurants and dental clinics.
- **Score distribution**: 0.7-0.8 343,244 | 0.8-0.9 382,115 | 0.9-1.0 1,065,178.
  The top decile *grew* by 13,671 versus the 0.50-cutoff run, which is the
  `brand_wikidata` identifier evidence promoting pairs that previously topped out
  lower.
- **Unnamed OSM POIs** cap at 0.80 because a null name scores a neutral 0.5. Sampling
  30 of them in 0.7-0.8 found ~83% correct (gas stations, churches, playgrounds and
  banks metres from a matching Overture name), so they survive the new cutoff.
  Renormalising the weights over components that actually have evidence — as the
  identifier component now does — would let them score on merit; not yet done.
- Overture `brand_wikidata` is populated on 159,623 rows (1.27%), falling to 83,543
  after internal dedup. Branded chains dedupe at ~48% against ~5% overall, which is
  itself a useful signal about Overture's source-merge duplication.
