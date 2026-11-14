# Switch rule (pre-registration 6.2)

Applied Nov 7, 2026 to the 200 exploratory human texts (100 HN, 100 SE), before any confirmatory analysis. Rule: presence is the primary measure if a tell occurs in under 5% of human texts or under 20 times in all; otherwise the rate. P6 is presence, P8 and P9 rate and P10 CV by definition. S8 tells are fixed by the pre-registration.

Inputs: `switch_rule_inputs.csv`. Result: `switch_rule.json`.

| Tell | Human share | Human total | Measure |
| --- | --- | --- | --- |
| P1 contrast frame | 4.5% | 9 | presence |
| P2 rather than | 2.5% | 7 | presence |
| P3 rule of three | 11.5% | 24 | rate |
| P4 staccato | 2.0% | 7 | presence |
| P5 signposting | 1.0% | 2 | presence |
| P6 summary closer | 1.5% | 3 | presence (fixed) |
| P7 worth noting | 1.0% | 2 | presence |
| P8 participial clauses | 8.5% | 18 | rate (fixed) |
| P9 nominalizations | 79% | 442 | rate (fixed) |
| P10 CV | | | CV (fixed) |

Every secondary count tell (S1-S7) falls under the presence rule. Every tell is still reported both ways in `descriptive.csv`.
