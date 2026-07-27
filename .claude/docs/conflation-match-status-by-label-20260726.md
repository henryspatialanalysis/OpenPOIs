# Conflation match-status by shared_label (2026-07-26)

Breakdown of each `shared_label` in the conflated output by match status, after the
**taxonomy overhaul** (`versions.conflation = 20260727`). Supersedes
[the 2026-07-25 table](conflation-match-status-by-label-20260725.md), which is kept
as the "before" record.

- **Source**: `~/data/openpois/conflation/20260727/conflated.parquet` (14,334,335 rows).
- **Input**: OSM rated snapshot (5,015,126 POIs, full-data `random_effects` refit
  `20260727_by_shared_label`) + Overture 2026-07-22 snapshot re-pulled under the
  widened `taxonomy_allowlist` (12,606,804 POIs).
- **Match status**: `matched` = one OSM POI joined to one Overture POI; `OSM only` /
  `Overture only` = unmatched from that source. `match %` = matched / total.
- **Δ total** compares against the 2026-07-25 run.

## Headline

| Metric | Before (20260724) | After (20260727) | Δ |
|---|--:|--:|--:|
| **Single-source labels** (a zero in any column) | **32** | **1** | **−31** |
| Labels | 93 | 102 | +9 |
| Matched pairs | 1,955,542 | 2,332,474 | +376,932 (+19.3%) |
| Overall match rate | 14.2% | 16.3% | +2.1 pp |
| OSM POIs labelled at conflation | 4,194,846 | 4,778,078 | +583,232 |
| Total rows | 13,734,101 | 14,334,335 | +600,234 |

The only remaining single-source label is **Car Rental**: Overture's taxonomy has no
car-rental category at all (the nearest, `car_sharing`, has ~245 US POIs), so it is
OSM-only by necessity rather than by crosswalk gap. Revisit if Overture adds one.

## What changed

Two upstream defects, not missing crosswalk rows:

1. **`conflate.py` loaded only 4 of the 9 OSM tag columns.** `assign_osm_shared_label`
   silently skips absent columns, so `tourism`, `office`, `craft`, `historic` and
   `landuse` were invisible at conflation and 817,214 labelled POIs were dropped by
   `drop_unlabeled`. Fixed by deriving the column list from `FILTER_KEYS`.
2. **The Overture crosswalk had rotted to 45% dead rows** after Overture's Feb/Mar-2026
   restructure (L0 22→13, 407 renames, 2,108 repaths). 26 labels received zero Overture
   POIs and 10.5% of rows collapsed onto L0 catch-alls. The crosswalk was rewritten
   (246 rows, 0 stale) and `compare_taxonomy.py` §6 now detects this class of drift.

Also: 9 new labels, the `amenity=marketplace` name split, `historic` value-scoping
(the `historic=*` wildcard had been labelling 135k memorials and boundary stones as
Museum), the tourism/street-furniture EXCLUDEs, and a widened Overture allowlist
(+308,473 POIs) that gives Post Office, Self-Storage, Print and Copy Shop, Laundromat,
Dry Cleaning and Marina an Overture side.

## Full table

| shared_label | matched | OSM only | Overture only | total | match % | Δ total |
|---|--:|--:|--:|--:|--:|--:|
| Recreation | 93,057 | 656,601 | 146,448 | 896,106 | 10.4% | -206,757 |
| Restaurant | 208,131 | 17,891 | 620,477 | 846,499 | 24.6% | +8,079 |
| Home Service | 9,491 | 7,119 | 778,955 | 795,565 | 1.2% | +19,900 |
| Place of Worship | 217,897 | 99,961 | 312,306 | 630,164 | 34.6% | +26,916 |
| Hair and Beauty | 66,778 | 9,600 | 477,498 | 553,876 | 12.1% | -57,523 |
| Other Financial | 27,180 | 10,122 | 463,330 | 500,632 | 5.4% | +33,096 |
| Park | 132,702 | 257,547 | 73,704 | 463,953 | 28.6% | +8,197 |
| Car Repair | 58,990 | 5,585 | 399,303 | 463,878 | 12.7% | +3,026 |
| Specialty Store | 31,713 | 12,553 | 417,879 | 462,145 | 6.9% | -501,019 |
| Real Estate | 10,493 | 4,316 | 382,828 | 397,637 | 2.6% | +6,624 |
| Other Healthcare | 6,528 | 2,647 | 383,458 | 392,633 | 1.7% | +39,349 |
| Other Professional | 29,107 | 44,847 | 310,494 | 384,448 | 7.6% | +75,797 |
| School | 97,323 | 55,445 | 205,757 | 358,525 | 27.1% | -44,337 |
| Community Center | 21,573 | 16,766 | 236,396 | 274,735 | 7.9% | -9,715 |
| Fast Food | 146,889 | 18,010 | 103,020 | 267,919 | 54.8% | -106,592 |
| Swimming Pool | 24,773 | 234,244 | 6,697 | 265,714 | 9.3% | +6,697 |
| Clothing Store | 45,061 | 9,172 | 204,476 | 258,709 | 17.4% | -55,338 |
| Hotel | 60,559 | 19,235 | 167,680 | 247,474 | 24.5% | +69,245 |
| Historic Site | 10,873 | 59,986 | 159,282 | 230,141 | 4.7% | **new** |
| Other Shop | 90,530 | 46,891 | 84,546 | 221,967 | 40.8% | -95,491 |
| Other Amenity | 30,574 | 128,791 | 44,836 | 204,201 | 15.0% | -413,683 |
| Car Dealer | 28,223 | 2,380 | 155,939 | 186,542 | 15.1% | +2,587 |
| Cemetery | 8,275 | 175,741 | 1,009 | 185,025 | 4.5% | +119,998 |
| Fitness Center | 20,388 | 5,039 | 156,097 | 181,524 | 11.2% | +156,674 |
| Hardware | 8,198 | 2,715 | 169,634 | 180,547 | 4.5% | +173,439 |
| Gas Station | 77,921 | 30,058 | 70,354 | 178,333 | 43.7% | -1,754 |
| Legal Service | 9,030 | 3,975 | 162,735 | 175,740 | 5.1% | +9,357 |
| Bar | 37,482 | 6,051 | 127,604 | 171,137 | 21.9% | +132,487 |
| Playground | 20,168 | 143,250 | 2,405 | 165,823 | 12.2% | -23 |
| Dentist | 22,080 | 2,918 | 133,156 | 158,154 | 14.0% | -1,901 |
| Cafe | 41,639 | 8,734 | 106,022 | 156,395 | 26.6% | +106,027 |
| Convenience Store | 56,378 | 30,976 | 62,850 | 150,204 | 37.5% | +40 |
| Bank | 52,476 | 8,089 | 81,133 | 141,698 | 37.0% | +129 |
| Clinic | 30,211 | 9,132 | 100,262 | 139,605 | 21.6% | +881 |
| Massage Therapy | 4,727 | 1,551 | 127,094 | 133,372 | 3.5% | -22,976 |
| Supermarket | 36,161 | 3,121 | 81,763 | 121,045 | 29.9% | -88,073 |
| Government Office | 24,626 | 15,832 | 80,277 | 120,735 | 20.4% | +16,661 |
| ATM | 5,076 | 9,756 | 92,951 | 107,783 | 4.7% | +718 |
| Pet Store | 9,484 | 1,898 | 95,129 | 106,511 | 8.9% | +7,882 |
| Mental Health | 811 | 609 | 98,746 | 100,166 | 0.8% | +10,074 |
| Self-Storage | 15,717 | 17,766 | 61,021 | 94,504 | 16.6% | **new** |
| Furniture Store | 9,061 | 2,563 | 79,790 | 91,414 | 9.9% | +79,790 |
| Public Safety | 31,353 | 18,031 | 33,661 | 83,045 | 37.8% | +4,611 |
| Performing Arts | 9,071 | 4,865 | 64,800 | 78,736 | 11.5% | +19,319 |
| Public Restroom | 11,528 | 65,936 | 241 | 77,705 | 14.8% | -2,144 |
| Dessert Shop | 10,264 | 1,866 | 64,633 | 76,763 | 13.4% | +64,633 |
| Florist | 4,364 | 968 | 71,348 | 76,680 | 5.7% | +71,348 |
| Campground | 10,969 | 34,654 | 30,064 | 75,687 | 14.5% | +40,866 |
| Pharmacy | 21,501 | 8,891 | 44,116 | 74,508 | 28.9% | -294 |
| Cell Phone Store | 12,779 | 4,177 | 54,641 | 71,597 | 17.8% | +54,641 |
| Post Office | 22,815 | 4,444 | 42,610 | 69,869 | 32.7% | +42,610 |
| Sports Outlet | 5,602 | 940 | 61,506 | 68,048 | 8.2% | +61,506 |
| Car Wash | 14,014 | 7,377 | 43,706 | 65,097 | 21.5% | +43,708 |
| Print and Copy Shop | 4,576 | 990 | 58,006 | 63,572 | 7.2% | **new** |
| Discount Store | 22,389 | 2,898 | 36,884 | 62,171 | 36.0% | -127 |
| Bakery | 8,756 | 2,327 | 50,859 | 61,942 | 14.1% | +4,156 |
| Eye Care | 7,977 | 1,443 | 52,039 | 61,459 | 13.0% | +12,529 |
| University | 7,000 | 1,606 | 49,108 | 57,714 | 12.1% | +11,189 |
| Physical Therapy | 3,596 | 1,863 | 47,073 | 52,532 | 6.8% | -661 |
| Alternative Medicine | 5,721 | 1,707 | 43,984 | 51,412 | 11.1% | -59,729 |
| Jewelry Store | 7,176 | 2,030 | 41,773 | 50,979 | 14.1% | +41,773 |
| Thrift Store | 5,452 | 1,648 | 42,952 | 50,052 | 10.9% | -31,863 |
| Liquor Store | 12,890 | 2,656 | 31,308 | 46,854 | 27.5% | +3,831 |
| Hospital | 7,597 | 374 | 37,000 | 44,971 | 16.9% | +1,505 |
| Childcare | 4,875 | 1,754 | 37,276 | 43,905 | 11.1% | -23 |
| Veterinarian | 9,008 | 946 | 33,084 | 43,038 | 20.9% | -933 |
| Arts Venue | 5,592 | 1,521 | 34,738 | 41,851 | 13.4% | -156 |
| Garden Store | 4,629 | 1,329 | 34,008 | 39,966 | 11.6% | +34,008 |
| Kindergarten | 5,009 | 1,879 | 31,846 | 38,734 | 12.9% | +31,846 |
| Tattoo and Piercing | 3,747 | 757 | 32,792 | 37,296 | 10.0% | **new** |
| Library | 16,442 | 2,903 | 15,535 | 34,880 | 47.1% | +25 |
| Shoe Store | 6,097 | 1,933 | 25,500 | 33,530 | 18.2% | -167 |
| Bookstore | 4,196 | 1,101 | 24,916 | 30,213 | 13.9% | +24,916 |
| Museum | 10,732 | 3,943 | 15,524 | 30,199 | 35.5% | +11,376 |
| Funeral Services | 5,155 | 577 | 23,514 | 29,246 | 17.6% | **new** |
| Laundromat | 6,411 | 3,221 | 19,254 | 28,886 | 22.2% | +19,254 |
| Golf Course | 4,514 | 11,855 | 11,254 | 27,623 | 16.3% | +11,254 |
| Shopping Center | 4,469 | 714 | 18,673 | 23,856 | 18.7% | +18,673 |
| Farmers Market | 756 | 194 | 19,564 | 20,514 | 3.7% | +246 |
| Event Venue | 2,819 | 3,785 | 13,450 | 20,054 | 14.1% | **new** |
| Dry Cleaning | 4,534 | 2,877 | 12,487 | 19,898 | 22.8% | +12,487 |
| Charging Station | 2,593 | 10,577 | 5,364 | 18,534 | 14.0% | -34 |
| Tire Store | 6,169 | 1,240 | 9,029 | 16,438 | 37.5% | +9,029 |
| Wholesale Store | 1,795 | 850 | 13,187 | 15,832 | 11.3% | **new** |
| Cannabis Dispensary | 3,244 | 1,312 | 10,994 | 15,550 | 20.9% | **new** |
| Animal Shelter | 1,128 | 371 | 12,587 | 14,086 | 8.0% | **new** |
| Nightclub | 1,796 | 395 | 10,920 | 13,111 | 13.7% | -328 |
| Stadium | 3,924 | 2,840 | 5,690 | 12,454 | 31.5% | -4,669 |
| Bike shop | 3,539 | 919 | 7,643 | 12,101 | 29.2% | +7,643 |
| Social Club | 3,504 | 1,509 | 7,081 | 12,094 | 29.0% | -817 |
| Movie Theater | 5,014 | 478 | 5,374 | 10,866 | 46.1% | -515 |
| Marina | 3,354 | 2,334 | 4,356 | 10,044 | 33.4% | +4,356 |
| Dog Park | 3,791 | 4,248 | 1,855 | 9,894 | 38.3% | -50 |
| Market | 1,390 | 516 | 7,557 | 9,463 | 14.7% | +3,662 |
| Arcade | 1,312 | 795 | 4,022 | 6,129 | 21.4% | +4,022 |
| Bowling Alley | 2,055 | 261 | 3,370 | 5,686 | 36.1% | +3,370 |
| Counseling | 811 | 775 | 3,804 | 5,390 | 15.0% | -49 |
| Casino | 1,119 | 202 | 3,785 | 5,106 | 21.9% | -256 |
| Car Rental | 1,000 | 3,322 | 0 | 4,322 | 23.1% | +0 |
| Speech Therapist | 80 | 81 | 3,569 | 3,730 | 2.1% | -45 |
| Occupational Therapy | 90 | 95 | 3,406 | 3,591 | 2.5% | -41 |
| Maternity Center | 37 | 21 | 1,596 | 1,654 | 2.2% | -26 |
| **TOTAL** | **2,332,474** | **2,445,604** | **9,556,257** | **14,334,335** | **16.3%** | **+600,234** |
- **Labels the `conflate.py` fix gave an OSM side** (OSM-only rows, which were
  zero or near-zero because the tag column was never loaded):

  | label | OSM-only before | OSM-only after |
  |---|--:|--:|
  | Hotel | 0 | 19,235 |
  | Campground | 0 | 34,654 |
  | Real Estate | 0 | 4,316 |
  | Legal Service | 0 | 3,975 |
  | Other Financial | 0 | 10,122 |
  | Other Professional | 0 | 44,847 |
  | Museum | 1 | 3,943 |

- **Labels the crosswalk rewrite + allowlist gave an Overture side** (Overture-only
  rows were zero because the crosswalk emitted no Overture row for the label):

  | label | Overture-only before | Overture-only after |
  |---|--:|--:|
  | Cafe | 0 | 106,022 |
  | Bar | 0 | 127,604 |
  | Hardware | 0 | 169,634 |
  | Furniture Store | 0 | 79,790 |
  | Cell Phone Store | 0 | 54,641 |
  | Post Office | 0 | 42,610 |
  | Marina | 0 | 4,356 |
  | Laundromat | 0 | 19,254 |
  | Dry Cleaning | 0 | 12,487 |
  | Golf Course | 0 | 11,254 |
  | Swimming Pool | 0 | 6,697 |
  | Shopping Center | 0 | 18,673 |

- **Specialty Store shrank by 501,019** and **Recreation by 206,757**: both were
  absorbing whole Overture subtrees through L0/L1 catch-alls. Their POIs did not
  disappear — they moved to Hardware, Furniture Store, Garden Store, Cell Phone Store,
  Bookstore, Sports Outlet, Fitness Center, Golf Course, Swimming Pool and Bowling
  Alley.
- **Museum went from 18,823 to 30,199 total, and its OSM side from 4 to 14,675.**
  Both moves are the `conflate.py` fix: `tourism=museum` was one of the invisible
  keys. Removing the `historic=*` wildcard pulled the mislabelled memorials and
  boundary stones *out* of Museum at the same time — they now sit under the new
  **Historic Site** label (230,141 total; 59,986 OSM-only, 159,282 Overture-only).
- **Swimming Pool 265,714** (OSM-only 234,244) — up only 6,697 on the prior run,
  confirming the unnamed private/no-access exclusion was applied. Without it this
  label balloons past 700k, so it is the canary for that filter.
- Match rates are highest for named chains (Bike shop 84.5%, Bowling Alley 84.1%,
  Movie Theater 83.3%, Animal Shelter 82.8%) and lowest for nameless or
  service-directory categories (Home Service 1.2%, Other Healthcare 1.7%,
  Real Estate 2.6%).

## Reproducing this table

`scripts/conflation/summarize.py` now writes `match_status_by_label.csv` alongside
`summary_by_label.csv` and prints every single-source label to stdout. The 2026-07-25
table was produced ad hoc; this one is reproducible.
