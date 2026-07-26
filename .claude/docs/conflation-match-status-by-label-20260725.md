# Conflation match-status by shared_label (2026-07-25)

Breakdown of each `shared_label` in the conflated output by match status, for the
**July 2026** refresh (`versions.conflation = 20260724`).

- **Source**: `~/data/openpois/conflation/20260724/conflated.parquet` (13,734,101 rows).
- **Input**: OSM rated snapshot **after** the unnamed private/no exclusion (5,015,126 OSM POIs;
  see the exclusion note in [data-sources.md](data-sources.md)) + Overture 2026-07-22 snapshot.
- **Match status**: `matched` = one OSM POI joined to one Overture POI; `OSM only` /
  `Overture only` = unmatched from that source. `match %` = matched / total for the label.

| shared_label | matched | OSM only | Overture only | total | match % |
|---|--:|--:|--:|--:|--:|
| Recreation | 67,707 | 670,879 | 364,277 | 1,102,863 | 6.1% |
| Specialty Store | 22,719 | 34,806 | 905,639 | 963,164 | 2.4% |
| Restaurant | 207,526 | 18,466 | 612,428 | 838,420 | 24.8% |
| Home Service | 587 | 188 | 774,890 | 775,665 | 0.1% |
| Other Amenity | 50,892 | 136,328 | 430,664 | 617,884 | 8.2% |
| Hair and Beauty | 66,836 | 9,542 | 535,021 | 611,399 | 10.9% |
| Place of Worship | 201,614 | 71,089 | 330,545 | 603,248 | 33.4% |
| Other Financial | 0 | 0 | 467,536 | 467,536 | 0.0% |
| Car Repair | 58,749 | 5,826 | 396,277 | 460,852 | 12.7% |
| Park | 97,127 | 293,122 | 65,507 | 455,756 | 21.3% |
| School | 91,806 | 53,374 | 257,682 | 402,862 | 22.8% |
| Real Estate | 0 | 0 | 391,013 | 391,013 | 0.0% |
| Fast Food | 145,233 | 16,006 | 213,272 | 374,511 | 38.8% |
| Other Healthcare | 6,363 | 2,811 | 344,110 | 353,284 | 1.8% |
| Other Shop | 132,229 | 73,275 | 111,954 | 317,458 | 41.7% |
| Clothing Store | 45,212 | 8,748 | 260,087 | 314,047 | 14.4% |
| Other Professional | 0 | 0 | 308,651 | 308,651 | 0.0% |
| Community Center | 8,823 | 5,285 | 270,342 | 284,450 | 3.1% |
| Swimming Pool | 10,548 | 248,469 | 0 | 259,017 | 4.1% |
| Supermarket | 38,045 | 3,882 | 167,191 | 209,118 | 18.2% |
| Car Dealer | 24,797 | 2,303 | 156,855 | 183,955 | 13.5% |
| Gas Station | 76,627 | 31,350 | 72,110 | 180,087 | 42.5% |
| Hotel | 0 | 0 | 178,229 | 178,229 | 0.0% |
| Legal Service | 0 | 0 | 166,383 | 166,383 | 0.0% |
| Playground | 10,543 | 152,875 | 2,428 | 165,846 | 6.4% |
| Dentist | 22,036 | 2,962 | 135,057 | 160,055 | 13.8% |
| Massage Therapy | 4,701 | 1,577 | 150,070 | 156,348 | 3.0% |
| Convenience Store | 52,168 | 33,798 | 64,198 | 150,164 | 34.7% |
| Bank | 52,268 | 8,296 | 81,005 | 141,569 | 36.9% |
| Clinic | 29,663 | 9,679 | 99,382 | 138,724 | 21.4% |
| Alternative Medicine | 5,818 | 1,609 | 103,714 | 111,141 | 5.2% |
| ATM | 4,534 | 10,298 | 92,233 | 107,065 | 4.2% |
| Government Office | 13,417 | 5,020 | 85,637 | 104,074 | 12.9% |
| Pet Store | 6,006 | 966 | 91,657 | 98,629 | 6.1% |
| Mental Health | 787 | 633 | 88,672 | 90,092 | 0.9% |
| Thrift Store | 3,373 | 1,171 | 77,371 | 81,915 | 4.1% |
| Public Restroom | 8,718 | 68,746 | 2,385 | 79,849 | 10.9% |
| Public Safety | 30,800 | 18,584 | 29,050 | 78,434 | 39.3% |
| Pharmacy | 21,286 | 9,105 | 44,411 | 74,802 | 28.5% |
| Cemetery | 3,496 | 60,171 | 1,360 | 65,027 | 5.4% |
| Discount Store | 22,342 | 2,945 | 37,011 | 62,298 | 35.9% |
| Performing Arts | 8,093 | 5,042 | 46,282 | 59,417 | 13.6% |
| Bakery | 8,120 | 1,923 | 47,743 | 57,786 | 14.1% |
| Physical Therapy | 3,550 | 1,909 | 47,734 | 53,193 | 6.7% |
| Cafe | 39,686 | 10,682 | 0 | 50,368 | 78.8% |
| Eye Care | 7,789 | 1,631 | 39,510 | 48,930 | 15.9% |
| University | 6,851 | 1,755 | 37,919 | 46,525 | 14.7% |
| Veterinarian | 8,948 | 1,005 | 34,018 | 43,971 | 20.3% |
| Childcare | 4,799 | 1,830 | 37,299 | 43,928 | 10.9% |
| Hospital | 7,566 | 405 | 35,495 | 43,466 | 17.4% |
| Liquor Store | 12,653 | 2,893 | 27,477 | 43,023 | 29.4% |
| Arts Venue | 2,207 | 675 | 39,125 | 42,007 | 5.3% |
| Bar | 31,644 | 7,006 | 0 | 38,650 | 81.9% |
| Library | 16,322 | 3,023 | 15,510 | 34,855 | 46.8% |
| Campground | 0 | 0 | 34,821 | 34,821 | 0.0% |
| Shoe Store | 6,056 | 1,974 | 25,667 | 33,697 | 18.0% |
| Post Office | 3,793 | 23,466 | 0 | 27,259 | 13.9% |
| Fitness Center | 16,445 | 5,847 | 2,558 | 24,850 | 66.2% |
| Car Wash | 12,835 | 8,554 | 0 | 21,389 | 60.0% |
| Farmers Market | 0 | 0 | 20,268 | 20,268 | 0.0% |
| Museum | 3 | 1 | 18,819 | 18,823 | 0.0% |
| Charging Station | 2,333 | 10,837 | 5,398 | 18,568 | 12.6% |
| Stadium | 3,630 | 3,134 | 10,359 | 17,123 | 21.2% |
| Cell Phone Store | 12,193 | 4,763 | 0 | 16,956 | 71.9% |
| Golf Course | 4,079 | 12,290 | 0 | 16,369 | 24.9% |
| Nightclub | 1,373 | 332 | 11,734 | 13,439 | 10.2% |
| Social Club | 3,463 | 1,550 | 7,898 | 12,911 | 26.8% |
| Dessert Shop | 9,815 | 2,315 | 0 | 12,130 | 80.9% |
| Furniture Store | 8,543 | 3,081 | 0 | 11,624 | 73.5% |
| Movie Theater | 4,964 | 527 | 5,890 | 11,381 | 43.6% |
| Dog Park | 3,196 | 4,843 | 1,905 | 9,944 | 32.1% |
| Laundromat | 1,448 | 8,184 | 0 | 9,632 | 15.0% |
| Jewelry Store | 6,488 | 2,718 | 0 | 9,206 | 70.5% |
| Dry Cleaning | 1,070 | 6,341 | 0 | 7,411 | 14.4% |
| Tire Store | 5,360 | 2,049 | 0 | 7,409 | 72.3% |
| Hardware | 5,617 | 1,491 | 0 | 7,108 | 79.0% |
| Kindergarten | 4,814 | 2,074 | 0 | 6,888 | 69.9% |
| Sports Outlet | 5,313 | 1,229 | 0 | 6,542 | 81.2% |
| Garden Store | 4,429 | 1,529 | 0 | 5,958 | 74.3% |
| Market | 1,538 | 1,318 | 2,945 | 5,801 | 26.5% |
| Marina | 1,295 | 4,393 | 0 | 5,688 | 22.8% |
| Counseling | 791 | 795 | 3,853 | 5,439 | 14.5% |
| Casino | 1,119 | 202 | 4,041 | 5,362 | 20.9% |
| Florist | 4,162 | 1,170 | 0 | 5,332 | 78.1% |
| Bookstore | 4,031 | 1,266 | 0 | 5,297 | 76.1% |
| Shopping Center | 4,241 | 942 | 0 | 5,183 | 81.8% |
| Bike shop | 3,402 | 1,056 | 0 | 4,458 | 76.3% |
| Car Rental | 750 | 3,572 | 0 | 4,322 | 17.4% |
| Speech Therapist | 80 | 81 | 3,614 | 3,775 | 2.1% |
| Occupational Therapy | 89 | 96 | 3,447 | 3,632 | 2.5% |
| Bowling Alley | 1,937 | 379 | 0 | 2,316 | 83.6% |
| Arcade | 1,186 | 921 | 0 | 2,107 | 56.3% |
| Maternity Center | 37 | 21 | 1,622 | 1,680 | 2.2% |
| **TOTAL** | **1,955,542** | **2,239,304** | **9,539,255** | **13,734,101** | **14.2%** |

## Notes

- Overall split: matched 14.2%, OSM-only 16.3%, Overture-only 69.5% — Overture contributes
  far more POIs, so most labels are Overture-dominated.
- **OSM-strong labels** (OSM-only much greater than Overture-only): Recreation, Park, Playground,
  Swimming Pool, Public Restroom, Cemetery, Post Office, Golf Course, Charging Station,
  Laundromat, Dry Cleaning, Marina — outdoor / civic / infrastructure features OSM maps well.
- **Overture-only labels** (0 matched, 0 OSM): Other Financial, Real Estate, Hotel, Legal
  Service, Other Professional, Campground, Farmers Market — the OSM crosswalk does not emit
  these labels, so they are entirely Overture-origin.
- **Zero-Overture labels** (Cafe, Bar, Dessert Shop, Furniture Store, Tire Store, Hardware, ...):
  Overture categorizes these under different branches, so the crosswalk never lands them on the
  label from the Overture side; all such rows are OSM-origin, inflating their match %.
- Highest match rates are named chains (Bowling Alley 84%, Bar 82%, Shopping Center 82%);
  lowest are nameless / single-source categories (Home Service 0.1%, Museum ~0%, Mental Health 0.9%).

