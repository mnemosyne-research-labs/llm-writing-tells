# Power simulation (pre-registration 6.6)

Run Nov 7, 2026, before the confirmatory analysis (`llm_tells.power`, 1,000 replicates per tell). Inputs from the exploratory split: human rates (per 1,000 words) or prevalence per register, extra-Poisson variance of human counts (half placed on the prompt, half on the text, an assumption), and observed word counts per source (Muse partly from human word counts, since only 10 exploratory Muse texts existed at the time). Simulated: 300 prompts per register, one human and three model texts each, true ratio 2 for every model; test: the pre-registered Poisson GEE contrast at α = 0.005 (the first Holm step).

Limitation: the simulation uses the GEE only. When a simulated cell has no occurrences, the GEE is skipped and the replicate counts as a miss, while the real analysis would switch to the sparse fallback (Haldane ratio, permutation test). Power for the rarest tells is therefore understated, but at prevalences of 1-2.5% in human text it is low either way.

| Tell | Measure | Human base, HN | Human base, SE | Power at ratio 2 | Estimate-only |
| --- | --- | --- | --- | --- | --- |
| p1_contrast | presence | 0.050 | 0.040 | 0.80 | yes |
| p2_rather_than | presence | 0.010 | 0.040 | 0.04 | yes |
| p3_tricolon | rate | 0.897 | 1.064 | 1.00 | no |
| p4_staccato | presence | 0.010 | 0.030 | 0.04 | yes |
| p5_signpost | presence | 0.020 | 0.005 | 0.00 | yes |
| p6_closer | presence | 0.020 | 0.010 | 0.03 | yes |
| p7_worth_noting | presence | 0.005 | 0.020 | 0.00 | yes |
| p8_participial | rate | 0.797 | 0.709 | 0.99 | no |
| p9_nominal | rate | 20.132 | 17.020 | 1.00 | no |

P10 (CV, Gaussian GEE, standardized difference 0.5) was not simulated: with about 200 or more texts per cell with at least four sentences, power for d = 0.5 at α = 0.005 is above 0.99 analytically.

**Reading the result:** P2, P4, P5, P6 and P7 are estimate-only at ratio 2. They can still be found to hold up if their true ratio is much larger than 2 (as Phase 1 suggests for signposting and summary closers); a null for any of them will be "inconclusive", never "does not hold up", unless its interval is narrow.
