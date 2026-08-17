# Focused Gemini preference-coherence replication

Valid records: **880**.

| Pair selection | Pairs | k=5 consistent | k=10 consistent | k=10 worse / improved | Exact p |
|---|---:|---:|---:|---:|---:|
| targeted | 8 | 6/8 | 3/8 | 3 / 0 | 0.2500 |
| confirmatory_random | 12 | 12/12 | 11/12 | 1 / 0 | 1.0000 |
| combined | 20 | 18/20 | 14/20 | 4 / 0 | 0.1250 |

| Attributes | Order-consistent pairs | First-position rate |
|---|---:|---:|
| 5 | 18/20 (90.0%) | 45.2% |
| 10 | 14/20 (70.0%) | 41.1% |

Paired exact McNemar p-value: **0.1250** (4 pairs worse at k=10; 0 worse at k=5).

The first-position rate changes by **-4.1%** from k=5 to k=10 across the 20 pairs (pair-level Wilcoxon **p=0.1950**; trial-level Fisher **p=0.2473**).

Mean actual thinking spend was **414.9** tokens at k=5 and **456.1** at k=10 despite the same 512-token cap.

## Pair-level results

| Pair | k | First-house win rate | P(edge > 0.5) | Order consistent | First-position rate |
|---|---:|---:|---:|---:|---:|
| BG | 5 | 21/22 (95.5%) | 100.0% | True | 45.5% |
| BG | 10 | 18/22 (81.8%) | 99.9% | True | 40.9% |
| BH | 5 | 14/22 (63.6%) | 89.5% | False | 31.8% |
| BH | 10 | 7/22 (31.8%) | 4.7% | False | 27.3% |
| CG | 5 | 16/22 (72.7%) | 98.3% | False | 22.7% |
| CG | 10 | 9/22 (40.9%) | 20.2% | False | 36.4% |
| DH | 5 | 18/22 (81.8%) | 99.9% | True | 40.9% |
| DH | 10 | 11/22 (50.0%) | 50.0% | False | 18.2% |
| EG | 5 | 20/22 (90.9%) | 100.0% | True | 40.9% |
| EG | 10 | 20/22 (90.9%) | 100.0% | True | 50.0% |
| EI | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| EI | 10 | 10/22 (45.5%) | 33.9% | False | 31.8% |
| HI | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| HI | 10 | 9/22 (40.9%) | 20.2% | False | 9.1% |
| GJ | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| GJ | 10 | 3/22 (13.6%) | 0.0% | True | 36.4% |
| AD | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| AD | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| DE | 5 | 17/22 (77.3%) | 99.5% | True | 27.3% |
| DE | 10 | 13/22 (59.1%) | 79.8% | False | 27.3% |
| AE | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| AE | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| IJ | 5 | 21/22 (95.5%) | 100.0% | True | 45.5% |
| IJ | 10 | 19/22 (86.4%) | 100.0% | True | 36.4% |
| FJ | 5 | 20/22 (90.9%) | 100.0% | True | 59.1% |
| FJ | 10 | 20/22 (90.9%) | 100.0% | True | 59.1% |
| AJ | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| AJ | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| CF | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| CF | 10 | 1/22 (4.5%) | 0.0% | True | 45.5% |
| AH | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| AH | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| DF | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| DF | 10 | 2/22 (9.1%) | 0.0% | True | 50.0% |
| AF | 5 | 1/22 (4.5%) | 0.0% | True | 54.5% |
| AF | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| CJ | 5 | 3/22 (13.6%) | 0.0% | True | 36.4% |
| CJ | 10 | 1/22 (4.5%) | 0.0% | True | 54.5% |
| EF | 5 | 0/22 (0.0%) | 0.0% | True | 50.0% |
| EF | 10 | 0/22 (0.0%) | 0.0% | True | 50.0% |

The targeted stratum is post-hoc; the 12-pair random stratum was selected before its calls were made. Neither covers all 45 pairs.
