# Pre-registration: which claimed tells of LLM-written text hold up (production study)

**Dated Nov 7, 2026, before any study data was collected.** Written to the OSF Preregistration template's headings (Bowman et al. 2020), with the secondary-data items of van den Akker et al. (2021) and the NLP items of van Miltenburg, van der Lee and Krahmer (2021). The design and the reasons for each choice are in [design.md](./design.md); its review log records the two-round external review (a Fable model acting as peer reviewer) that shaped this document. Any change after this date is listed in [Deviations](#deviations) with its date and reason, and reported in the post.

Data seen before registration: Phase 1 read keyword-pulled HN comments from 2023 to 2026 (`data/hn/`), none from the 2019 to 2022 frame used here. Feasibility checks before registration touched no study text: model throughput on a fixed test prompt, the Wikinews frame size (category counts only), and the HN item-ID range for the date windows.

## 1. Study information

**Question.** For each claimed tell on a fixed list, how much more often does it occur in text written by current open-weights instruction-tuned models than in comparable human text written before Nov 30, 2022, and how often does it occur in that human text (its false-positive rate)?

**Scope of the claim.** Three open-weights instruction-tuned models run locally (Qwen 3.8 27B, Gemma 4 E4B, Meta Muse Glimmer 30B), in two reply registers (Hacker News comment replies and Stack Exchange answers), in English. Not "LLMs" in general, not frontier chat products, not essays or news. A small Claude arm and a base-versus-instruct arm are exploratory.

**Hypotheses.** For each primary tell P1 to P9: the tell occurs at a higher rate (or in a larger share of texts, per the switch rule in 6.2) in model text than in human text (RR > 1). For P10: model texts have a lower coefficient of variation of sentence length than human texts. The smallest effect of interest is a rate or risk ratio of 2 (P10: a standardized difference of 0.5).

## 2. Design

Same-prompt matched corpora. Each human text is a reply to something: an HN comment replies to a parent comment under a story; a Stack Exchange answer replies to a question. Each model receives the same thing the human saw (story title and parent comment, or site, question title and question body) and is asked for a reply of about the human text's length. A **prompt** (one parent or one question) is the cluster: it has one human text and one text per model.

| Arm | Status | Texts |
| --- | --- | --- |
| Main: human versus three instruct models, two registers | Confirmatory (primary and secondary tests) | 800 prompts × (1 human + 3 models) = 3,200 |
| Claude: Claude Sonnet via Claude Code subagents, 50 HN and 50 SE prompts from the confirmatory split | Exploratory, descriptive only | 100 |
| Continuation: Llama 3.1 8B base versus Llama 3.1 8B Instruct versus the real human continuation, 150 long SE answers | Exploratory (tested with BH within the arm) | 450 |
| Recent HN: 200 replies from Jan 1, 2025 to Oct 31, 2026 | Exploratory, descriptive only; labelled possibly LLM-contaminated | 200 |

**Split corpus (Mak 2024).** Prompts are assigned to the exploratory split (25%, 100 per register) or the confirmatory split (75%, 300 per register) by a seeded hash of the prompt ID, before any text is generated. The exploratory split is used only to: fix bugs in the counters and finalize regular expressions, pilot the codebook, estimate rates for the power simulation, apply the switch rule (6.2) and flag estimate-only tells. No hypothesis is added or dropped on the basis of the exploratory split. Confirmatory analysis is run once, after the counters are frozen and committed to the study folder with a date.

## 3. Sampling plan

### 3.1 Human texts

**HN replies.** Frame: HN items with IDs 18,797,929 to 33,795,048 (created Jan 1, 2019 to Nov 29, 2022, checked against item timestamps). Item IDs drawn uniformly at random (seeded) and fetched from the official HN Firebase API until 400 eligible replies are found. Eligible: type comment; not dead or deleted; parent is a comment (not a story); created before Nov 30, 2022; after cleaning (section 4), 40 to 400 words; no code block (`<pre>`); quoted lines (starting with `>`) under half of the words; English (at least 90% of letters ASCII and at least 20% of words from a fixed English function-word list); parent comment at least 10 words. The story title is found by walking up the parent chain. HN comments cannot be edited after about two hours, so the creation date is the text's date.

**Stack Exchange answers.** Frame: answers created Jan 1, 2019 to Nov 29, 2022 on ten sites, 40 per site: English Language and Usage, Travel, Seasoned Advice (cooking), Personal Finance and Money, Parenting, The Workplace, Academia, History, Home Improvement (DIY), Bicycles. Fetched from the public Stack Exchange API without a key, from randomly placed (seeded) date windows within the frame per site, then sampled at random from the pooled candidates. Eligible: score at least 1 (no accepted-answer filter); **last edit, if any, before Nov 30, 2022** (answers edited later are excluded and counted); after cleaning, 40 to 600 words; no code block; English (as above); question body and title present. The question body is capped at its first 300 words for the prompt.

**Continuation sample.** 150 SE answers (15 per site) from the same frame and rules, but at least 300 words after cleaning, drawn **before** the main SE sample and excluded from it, so the two are disjoint.

**Recent HN sample.** 200 HN replies by the HN rules above, with IDs in the range for Jan 1, 2025 to Oct 31, 2026. Human only; no generation.

**Stopping rule.** Sampling stops when each quota is filled. Excluded items are counted by reason and reported.

### 3.2 Model texts

| Model | File | Quantization | Sampling | Thinking |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B (`qwen/qwen3.8-27b`) | lmstudio-community Qwen3.8-27B-MLX-4bit | 4-bit MLX | temperature 1.0, top-p 0.95, top-k 20 (its `generation_config.json`) | off (`reasoning_effort: none`) |
| Gemma 4 E4B (`google/gemma-4-e4b`) | lmstudio-community gemma-4-E4B-it-MLX-4bit | 4-bit MLX | temperature 1.0, top-p 0.95, top-k 64 (its `generation_config.json`) | off |
| Muse Glimmer 30B (`meta/muse-glimmer`) | lmstudio-community Muse-Glimmer-30B Q4_K_M GGUF | Q4_K_M | LM Studio defaults (temperature 0.8, top-p 0.95, top-k 40), since no generation config ships with the file | off if the model supports it; recorded |
| Llama 3.1 8B base and Instruct (continuation arm) | mlx-community Meta-Llama-3.1-8B-8bit and -Instruct-8bit | 8-bit MLX | temperature 0.6, top-p 0.9 (Meta's published generation config) | none |
| Claude Sonnet (Claude arm) | Claude Code subagent, `model: sonnet` | n/a | not controllable | n/a |

Each model file's SHA-256, LM Studio version, context length (8,192), parallel slots (4), seed (derived from prompt ID and model name) and `finish_reason` are recorded per text. One sample per prompt per model. No system prompt for local models. `max_tokens` is four times the target word count times 1.4, at most 2,500. Generation runs one model at a time, four requests in parallel.

**Prompts, verbatim** (`{…}` filled per item):

HN reply:
```
Here is a comment from a Hacker News discussion of the story "{story_title}".

{parent_text}

Write a reply to this comment, about {n_words} words long. Give only the text of the reply.
```

SE answer:
```
Here is a question posted on {site_name} Stack Exchange.

Title: {question_title}

{question_body}

Write an answer to this question, about {n_words} words long. Give only the text of the answer.
```

Continuation, base model (raw completion, no chat template): the question title, a blank line, then the first half of the human answer verbatim; the model continues.

Continuation, Instruct model (chat):
```
Continue the following text. Write about {n_words} more words, in the same style. Give only the continuation.

{first_half}
```
The split is at the sentence boundary nearest the middle by word count, at least 150 words on each side; `n_words` is the length of the real second half. Base-model output is cut at the first sentence boundary after `n_words` words (base models do not stop), and anything after a line that starts a new document (a new title, `Question:`, `Answer:` or similar) is removed; this rule applies to the base model only and is reported.

Claude arm: each Sonnet subagent receives 10 prompts, verbatim as above, with an instruction to answer each one independently, as if it were the only prompt, and to return them in a fixed JSON shape. The batch size and the uncontrolled system prompt and sampling are stated as limitations.

**Refusals and failures.** A model text that is empty, a refusal (a fixed refusal-phrase list, checked by hand on hits) or not English is regenerated once with a new seed; if it fails again it is excluded and counted. Truncated texts (`finish_reason` = `length`) are kept for every tell except the end-position tells (P6, S6 and S7), from which they are excluded.

## 4. Variables: cleaning and the primary tells

**Cleaning, identical for both sides.**
- HN text: HTML entities unescaped, `<p>` to paragraph breaks, other tags removed with their text kept, `<pre>` blocks removed, quoted lines (starting with `>`) removed, URLs removed.
- Stack Exchange text and all model text: rendered from Markdown with markdown-it-py (CommonMark) and walked as tokens. Code blocks, inline code, images, block quotes and URLs are removed; link text is kept. Headings and list items become their own lines; emphasis markers disappear. Formatting tells (S8) are counted from the same token stream, so both sides are counted from one rendering.
- Words: tokens matching `[A-Za-z0-9]+(?:['’][A-Za-z]+)*`. Apostrophes and quotes are normalized to straight ones before matching.
- Sentences: each paragraph or line is parsed separately with spaCy `en_core_web_sm` 3.8.0, and its sentence boundaries are used. Heading lines and list-item lines are marked and excluded from P4 and P10.

All counts are per text. Rates are per 1,000 words.

| ID | Tell (claimed form) | Operational definition (counter) | Primary measure |
| --- | --- | --- | --- |
| P1 | Contrast frame: "it's not X, it's Y"; "not just X but Y"; "not X, but Y" | Within a sentence: (a) a pronoun or "there" + negated copula (`'s not`, `isn't`, `is not`, `aren't`, `wasn't`…) followed within 80 characters, after a comma, semicolon, colon or dash, by a pronoun + copula; (b) `not (just|only|merely|simply) … but` where `but` is not followed by `also`; (c) `not … , but` within one clause span of 60 characters. Across sentences: a sentence containing a negated copula with a pronoun subject followed by a sentence that starts with `it's`, `it is`, `this is`, `that's` or `they're`. "Not only … but also" is counted separately and reported, not tested | per the switch rule |
| P2 | "Y rather than X" | `\brather than\b` | per the switch rule |
| P3 | Rule of three | Coordination groups with exactly three conjuncts: a token with exactly two `conj` children and a `cc` child (spaCy dependency parse) | per the switch rule |
| P4 | Staccato fragments | Runs of two or more consecutive prose sentences in one paragraph, each of five words or fewer; count of runs | per the switch rule |
| P5 | Signposting | Fixed phrase list, case-insensitive: here's the thing, here is the thing, here's why, here is why, here's the kicker, the key insight, the real question is, the bottom line, let's dive in, let's dive into, let's delve, let that sink in, the truth is, make no mistake, spoiler: | per the switch rule |
| P6 | Summary closer | The first sentence of the last paragraph, or either of the last two sentences, starts with: in conclusion, in summary, to summarize, to sum up, overall, ultimately, in short, all in all, in the end, the takeaway, key takeaway, the key takeaway, tl;dr, tldr. Texts of one sentence are scored 0. Truncated texts excluded | presence (end-position tell, at most once per text) |
| P7 | "It's worth noting" family | `(it's |it is )?(worth (noting|mentioning|pointing out)|important to (note|remember|understand|keep in mind))`, plus sentence-initial `note that` | per the switch rule |
| P8 | Present participial clauses (Reinhart anchor) | Tokens tagged `VBG` with dependency `advcl`, with no subordinator child (`mark`), and either preceded by a comma or first in the sentence | rate |
| P9 | Nominalizations (Reinhart anchor) | Tokens with POS NOUN, at least seven letters, ending in -tion(s), -ment(s), -ness(es), -ity or -ities (Biber 1988), minus a fixed list of non-derived nouns: city, cities, pity, nation(s), station(s), ration(s), cement, comment(s), moment(s), element(s), apartment(s), department(s), segment(s), garment(s), document(s) | rate |
| P10 | Uniform sentence length | Coefficient of variation (SD over mean) of words per prose sentence; texts with at least four prose sentences only | CV (continuous) |

**Secondary tells** (Benjamini-Hochberg across all secondary tests, q = 0.05):

| ID | Tell | Definition | Register |
| --- | --- | --- | --- |
| S1 | Em dash family | `—`, a spaced `–`, or `--` between words, as one class; numeric ranges excluded | both |
| S2 | Stacked transitions | Sentence-initial moreover, furthermore, additionally, in addition, consequently, notably, importantly | both |
| S3 | Flagged vocabulary composite | Word forms of: delve, tapestry, testament, realm, crucial, pivotal, leverage, seamless, robust, landscape, underscore, showcase, intricate, meticulous, commendable, camaraderie, nuanced, foster, multifaceted, navigate | both |
| S4 | 2026 vocabulary | quiet, quietly, load-bearing, genuinely, legible | both |
| S5 | Self-answered question | A sentence ending in `?` followed by a sentence of at most four words starting with yes, no, absolutely, not really, of course, sure, exactly, nope, definitely, probably, maybe or "the answer" | both |
| S6 | Sycophantic or chatbot opener | Text starts with: great question, good question, you're right, you're absolutely right, absolutely, certainly, of course, i understand, that's a great point, that's a fair point, totally, sure. Truncated texts not excluded (start-position) | both |
| S7 | Hedging closer | Last sentence starts with however, that said, that being said, of course, still. Truncated texts excluded | both |
| S8 | Formatting (a) any list, (b) any heading, (c) bold spans per 1,000 words, (d) a bold lead-in on list items (at least two list items that start with bold text) | From the markdown-it token stream | SE only |
| S9 | Content tells, model-coded: (a) generic, no first-hand or concrete specifics; (b) hype or grandiose framing | Binary per text, codebook in `coding/codebook.md` | both, coded subsample |

Descriptive only, not tested: emoji count; "not only … but also"; each word in S3 and S4 separately; per-model and per-register estimates of every tell (reported with BH-adjusted q-values as their own family).

## 5. Reliability and validation of the counters

The research standard's "checked by someone else" applies to the counters. With no paid human coder, the check is two model coders from different families, blind to each other and to source:
- **Coder A:** GLM-4.7-Flash (6-bit MLX, local), temperature 0, thinking off.
- **Coder B:** Claude Haiku via Claude Code subagents, batches of 25 texts; 10% of items recoded by a second Haiku run to measure its own stability.

Coders see cleaned text only (Markdown and source-revealing formatting removed), shuffled, under opaque IDs. The codebook gives each judgment tell's definition and examples; it is piloted on 20 exploratory texts and frozen before the validation sample is coded.

**Validation sample, sized per tell.** For each of P1 to P8 and S5 to S7, at least 30 counter-positive and 30 counter-negative texts from the confirmatory split, stratified by source and register, overlap allowed (about 300 to 400 texts in all). Each coder counts every coded tell in every text. Reported per tell: Cohen's kappa between coders (text-level presence), positive agreement, Krippendorff's alpha, and the counter's precision and recall against the texts where the coders agree, **separately for human and model texts** (differential error is the threat to the rate ratio). Kappa between two models is reported as instrument agreement, not as a human check.

**Thresholds.** If coder kappa for a tell is below 0.6, its precision and recall are reported but not relied on. If the counter's precision is below 0.7 or recall below 0.5 against coder consensus, or precision differs between human and model texts by more than 0.2, the tell's verdict carries "measurement not validated". The verdict is still reported; nothing is dropped.

**S9.** Coder A codes 60 texts per source and register cell (480); coder B codes 150 of them. Kappa reported; the rate comparison uses coder A, with coder B's estimate alongside.

**Optional human check.** The founder may hand-code 100 texts from the validation sample (`coding/human_sheet.csv`). If done, it is reported as a v2 addendum with model-versus-human kappa. The study does not wait for it.

## 6. Analysis plan

### 6.1 Models

Unit of analysis: the text. Cluster: the prompt (one parent or question). All estimates are population-averaged.

- **Rates (counts):** Poisson GEE, log link, offset log(words / 1,000), fixed effects source × register (source: human, Qwen, Gemma, Muse), exchangeable working correlation within prompt, robust sandwich standard errors. NB-family GEE is a sensitivity analysis.
- **Presence:** modified Poisson GEE (Zou 2004) on 0 or 1, the same fixed effects plus log word count as a covariate, giving risk ratios. Presence shares are reported at the human median word count per register (reported).
- **P10 (CV):** Gaussian GEE, same fixed effects; effect is the mean difference standardized by the human SD.
- **Stack Exchange site:** for SE-only estimates, site is a fixed covariate; the cluster bootstrap is stratified by site.

**Primary estimand per tell:** the average over the three models and the two registers (equal weights) of the log ratio between model and human: L = (1/6) Σ_r Σ_m (β_{m,r} − β_{human,r}). exp(L) is the average rate or risk ratio. For P10, the same average of standardized differences. Wald test of L = 0, two-sided, with 95% and 90% intervals. The register × source interaction is secondary.

**Sparse tells.** If a tell has fewer than 5 human occurrences in the confirmatory split, or the GEE fails to converge, the ratio is estimated as a Haldane-corrected ratio of rates (0.5 added to each group's count) with a 95% cluster-bootstrap interval (2,000 resamples of prompts within register), and the test is a within-prompt permutation test of source labels (5,000 permutations). The one-sided 95% bound is reported.

### 6.2 Switch rule (rate or presence)

Decided per tell on the exploratory split (200 human texts) before the confirmatory data is analyzed: if the tell occurs in fewer than 5% of human texts, or fewer than 20 times in all human texts, its primary measure is presence; otherwise the rate. P6 is presence, P8 and P9 rate and P10 CV by definition. The choices are recorded, dated, in `analysis/switch_rule.md` before the confirmatory run. Because 0 to 2 human occurrences in 100 texts per register say little, the rule is conservative and every tell is reported both ways.

### 6.3 Multiple comparisons

Primary: ten tests (P1 to P10), Holm correction, family-wise α = 0.05. Secondary: S1 to S9 with Benjamini-Hochberg, q = 0.05, as one family. Per-model and per-register estimates, the continuation arm and the classifier are their own BH families. The Claude arm and the recent HN sample are descriptive and untested.

### 6.4 False-positive rates and usefulness

For each tell, per register and pooled (equal register weights): the hit rate (share of model texts with at least one occurrence), the false-positive rate (share of human texts with at least one), and LR+ = hit rate / false-positive rate, with 95% cluster-bootstrap intervals; also at the thresholds "at least two" and "rate above the human 90th percentile". For P10: hit = CV below the human 25th percentile. Positive predictive value is shown only as an illustration at stated base rates (10% and 50% model text).

### 6.5 Verdicts

Assigned per primary tell, in this order:
1. **Holds up:** Holm-adjusted p < 0.05 in the hypothesized direction, point ratio ≥ 2 (P10: standardized difference ≤ −0.5) and point LR+ ≥ 2.
2. **Does not hold up:** the 90% interval of the ratio lies inside 0.67 to 1.5 (P10: inside −0.2 to 0.2), whether or not the difference is significant.
3. **Real but weak:** Holm-adjusted p < 0.05 in the hypothesized direction, but not meeting 1.
4. **Reversed:** Holm-adjusted p < 0.05 in the opposite direction.
5. **Inconclusive:** none of the above. A null without equivalence is reported as inconclusive, never as "no effect".

Secondary tells get the same verdicts with BH q-values in place of Holm. Tells flagged estimate-only by the simulation (6.6) are expected to be inconclusive if their true ratio is below about 3; that is stated with the result.

### 6.6 Power

Fixed by the budget at 300 confirmatory prompts per register. Before the confirmatory run, a simulation (1,000 replicates per tell, using exploratory human rates and overdispersion, cluster structure and the 3-to-1 model-to-human ratio) estimates power for ratio 2 at the Holm-adjusted α for the first test (0.005). Tells with power below 0.8 are flagged estimate-only. The simulation and its output are saved in `analysis/power.md`.

### 6.7 Discrimination (secondary)

Per-tell AUC (rate as score) with group-bootstrap intervals. A logistic regression on all primary and secondary tell features (S8 and S9 excluded, since they are not available for every text), with folds grouped by prompt (5-fold), balanced accuracy and AUC. A stylometric ceiling: a character 2-to-4-gram TF-IDF logistic regression on the cleaned, normalized text (dashes and quotes normalized), same folds. Reported: how much of the ceiling's separation the named tells reach. Human is the minority class (one human to three model texts per prompt); balanced accuracy is reported for that reason.

### 6.8 Exploratory analyses

- Continuation arm: each primary tell in Llama 3.1 8B base, Instruct and the real continuation; Instruct versus base and each versus human, Poisson GEE clustered by item, BH within the arm. The question is whether post-training introduces the tell (Reinhart et al. 2025).
- Claude arm: rates, presence and LR+ against the matched human texts; no tests.
- Recent HN (2025-2026): the false-positive rate of each tell, against the 2019-2022 rate; descriptive, possibly contaminated.
- Robustness: NB-GEE; cluster bootstrap for every primary tell; P1 with and without the cross-sentence rule; P6 with the last paragraph only; the HN length floor at 80 words.

## 7. Deviations policy

Anything changed after this date (a definition, a sample rule, a model, an analysis) is added to the table below with its date and reason, and every deviation is reported in the post. Bug fixes to counters found on the exploratory split are not deviations if they bring the code in line with the definitions above; changes to the definitions are.

## Deviations

| Date | What changed | Why |
| --- | --- | --- |
| Nov 7, 2026 | D1. Muse Glimmer runs with one system message, "Reasoning strength: none.", and 2,048 extra `max_tokens` for its hidden reasoning, which is stripped (returned separately by LM Studio and never counted) | Found in the pilot: Muse cannot switch reasoning off (`reasoning_effort` and template arguments are ignored); with its default ("high") it spent the whole token cap reasoning and returned empty text. "none" is the lowest level its chat template accepts; the system message is the template's own control and carries no style instruction. Qwen and Gemma produced no reasoning with thinking off |
| Nov 7, 2026 | D2. The continuation arm has 125 items, not 150 | 9 long answers shared a question with another (D3) and were dropped; 16 of the remaining 141 failed the pre-registered rule of at least 150 words on each side of the split; no replacements drawn |
| Nov 7, 2026 | D3. Stack Exchange: one answer per question, and no question in both the main and continuation samples. 59 main-arm answers were dropped (33 duplicate questions, 26 questions also in the continuation sample) and replaced by a new draw under the same rules (seed 20261109, `data/raw/se_2019_2022_supplement/`), keeping 40 per site. Every existing prompt kept its split; new prompts filled each site's exploratory quota in hash order. The Claude subset lost 3 SE prompts (47 instead of 50) | Found in the code review: the pre-registered cluster is the question, but the sampler drew answers, so 35 of 400 answers shared a question and two answers to one question could fall in different splits |
| Nov 7, 2026 | D4. Counter definitions refined after the code review, before any confirmatory data was analyzed: P1 drops "which" from the pronoun list; P1 rule (c) is token-based ("not" not preceded by do-support or a modal, not followed by a verb or "sure", not after "I'm/I am"); the cross-sentence rule does not count a second sentence that starts "it's worth" or "this is because"; P3 counts conjunct chains (spaCy links conjuncts in a chain, not as siblings) of exactly three items of the same part-of-speech group with exactly one "and" or "or"; P6 "overall" and "ultimately", and S7 "still", no longer require a following comma (the commas in section 4 were list separators); headings are excluded only from P4 and P10, as written; the em dash class excludes `--` and spaced en dashes between digits; Markdown is parsed with raw HTML shown as text (tags then stripped) and tables enabled, with table cells treated like list items | Reviewer tests on the exploratory human texts: rule (c) matched ordinary verb negation ("I did not like it, but…"), 9 of 12 human hits; P3 counted clause chains joined by "but"; raw HTML blocks lost their text on the human side only; tables became junk sentences on the model side only |
| Nov 7, 2026 | D6. For model output, the "not English" regeneration trigger uses only the ASCII-letter share (at least 90%), not the function-word share. Texts regenerated or excluded under the old trigger get their first-attempt sample back (same seed, re-requested) when it passes; otherwise the second attempt is kept and re-checked (`llm_tells.regen_fix`; the restored texts are in `data/generated/d6_patch.jsonl`, applied over the original records, which stay unchanged) | Gemma's first pass regenerated 41 of 800 texts and excluded 13, all plainly English: dense, function-word-poor prose fell under the 20% function-word share designed for human text. Keeping that rule would have removed the densest model texts, biasing against nominalization and other density tells. The human eligibility rule is unchanged |
| Nov 7, 2026 | D7. The continuation arm uses the same measure per tell as the main arm (the switch rule's rate or presence), not a Poisson rate GEE for every tell; for Instruct versus base, the sparse-fallback threshold counts the base model's occurrences | Decided before the confirmatory run, on the pre-run analysis review: consistent with the main arm, and the rare tells are zero-heavy in 125 items |
| Nov 7, 2026 | D8. The validation sample has fewer than 30 counter-positive texts for the rarest tells (P5: 5, S5: 6, P7: 9, S6: 25, S7: 28; every positive in the sample frame was taken). It was drawn per source-and-register cell (up to six positives and six negatives per cell and tell, each cell seeded on its own) so that Muse's cells could be added after its run without changing the others | Rare tells; the per-cell draw let coding start before generation ended |
| Nov 7, 2026 | D9. Coder A (GLM-4.7-Flash, thinking off) proved degenerate on several codes (0 of 275 texts with a tricolon or a participial clause). The registered check is reported as it is. Added, unregistered and labelled as such: the counters against Claude Haiku alone, and a GLM pass with reasoning on for five codes | Found when the coding came back; the registered rule (kappa below 0.6: precision and recall not relied on) is applied unchanged |
| Nov 7, 2026 | D5. HN parent comments in the prompt are capped at 300 words (11 of 400 prompts), as Stack Exchange question bodies are | Not stated in section 3.1 for HN; applied from the start |
| Nov 8, 2026 | D10. The optional human check is redrawn and narrowed, before any hand-coding. Instead of 100 random validation texts and 13 codes, the founder codes P3 and P8 only, on 121 confirmatory texts (`coding/human_sheet_v2.csv`, seed 20261117, `llm_tells.coding human-sheet`): for each code and side (human, model), 25 counter-positive texts; per side, 15 texts negative on both counters; texts Haiku already coded are taken first, so founder and Haiku can be compared (101 of 121). **Hypothesis and pass rule** (the section 5 thresholds, with the founder as reference): for each of P3 and P8, counter precision at least 0.7 on human and on model text, a human-model precision gap of at most 0.2, and recall at least 0.5 (weighted to the pool's counter-negative count; approximate). If met, the tell keeps "measurement not validated (registered check)" and adds "validated against one human coder (D10)". If not, the label stays and the rate ratio is also reported corrected for counter error (ratio times (precision / recall) on model text over the same on human text); if that corrected ratio is below 2, the post no longer lists the tell under "holds up" and reports it as doubtful on measurement. One coder, not blind to the hypotheses or results; texts blind to source (cleaned text, opaque IDs) | The random sheet held only 6 human-text rule-of-three and 4 human-text participial positives, too few to test the weakness Result 5 found (human-text precision 0.71 and 0.62 against Haiku). The other headline tells (P2, S1, S3) are string matches. Reported as an unregistered addendum |
| Nov 8, 2026 | D11. Added before any hand-coding: Claude Fable codes the same 121 D10 texts as a further coder, with the frozen v1 tell codebook (all eleven codes, as Haiku did), cleaned text, opaque IDs and batches of 25, through Claude Code subagents (this workspace's instructions in context) (`coding/fable_tells_v2/`). Its output is kept from the founder until her sheet is saved. Secondary and descriptive only: reported as counter-versus-Fable and founder-versus-Fable agreement for P3 and P8 (`coding/human_vs_fable.csv`, `coding/counter_vs_fable.csv`); the D10 pass rule stays on the founder's coding alone. Fable, Haiku and the Claude arm's generator share a vendor, so their errors may correlate | A check on the single, non-blind human coder: where founder and Fable disagree, both readings are reported; where they agree against Haiku, the Haiku supplement (D9) is weaker than it looked. Opus considered and not used: a second strong Claude model adds little over one |
| Nov 8, 2026 | D12. The founder coded 80 of the 121 D10 texts (sheet rows 1-57 and 99-121; the rows are in opaque-ID order, so the subset is close to random) and stopped, decided before any scoring. Counter-positive texts coded: P3 16 human, 17 model; P8 18 human, 16 model (planned 25 each); counter-negative 19-28 per side. The D10 pass rule is applied unchanged to these 80; recall is weighted as in D10. After row 19 the founder received written notes elaborating the codebook's P3 and P8 definitions, with judgment calls to flag (`coding/human_coder_notes.md`); the models had the codebook only. One judgment note (row 1) moved to a `note` column | Founder's time; each precision estimate is less certain (about plus or minus 0.2 instead of 0.17) |
| Nov 9, 2026 | D13. Adjudication of the founder's D12 codes, before any review. Items: every text and code (P3, P8) among the 80 coded texts where the founder's count differs from the counter's or Fable's (52 items: P3 23, P8 29), plus, per code, 5 texts drawn at random (seed 20261109) where all three agree, to estimate the errors all three share; 62 items on 49 texts (`coding/adjudication_sheet.csv`, key `adjudication_sheet.parquet`, `llm_tells.coding adjudication-sheet`). The founder sees the cleaned text, the distinct counts sorted and unlabelled, and the phrases the counter matched (`llm_tells.counters.matched_phrases`; counts unchanged, checked on all 3,297 texts); she enters a final count, which need not be one of the options, and quotes each instance. **Hypothesis and pass rule:** the D10 rule and consequence, unchanged, applied to the adjudicated codes (reviewed items take the final count; all others keep the first pass); expected: P8 still fails, P3 may pass. The adjudicated scores govern the verdicts in results.md and the post; the D12 first-pass scores are reported beside them. Also reported: per code and stratum, how many counts changed and whether the final count equals the counter's, Fable's or neither (`coding/adjudication_summary.csv`) | The founder's first pass can be wrong too. Reviewing only disagreements with the counter can move codes only toward the counter, which inflates its accuracy (discrepant analysis: Hadgu 1996; Miller 1998); Fable as a second trigger and the all-agree sample limit that. Not fully blind: the phrases reveal the counter's count, and an all-agree item shows one option. The all-agree sample is small (5 per code): it detects only a large shared error rate. Same coder, not blind to the hypotheses or the first-pass results |
| Nov 9, 2026 | D14. Exploratory frontier arm, added after the results and before any request: `gemini-3.8-flash` through the Gemini API free tier (OpenAI-compatible endpoint), on the same 800 prompts, verbatim, no system prompt, one sample each. Thinking cannot be switched off on this model; `reasoning_effort` is set to `low`, its lowest level (the main arm had thinking off or minimal, D1). Sampling at Google's defaults (no temperature or top-p sent); no seed (the endpoint rejects the field, found on the first test request, Nov 9, before any text was generated); max tokens = the registered cap plus 2,048 for thinking; the same regeneration rule (D6). A 20-prompt pilot on the exploratory split first, for tokens, refusals and rate limits only; its texts are kept. Measured with the frozen counters. **Hypothesis and pass rule:** each of the four tells listed as holding up (P2, S1, S3, P3) is also more common in Gemini than in the matched human texts. For each tell, on the confirmatory split pooled over both registers, Gemini against human with the main arm's per-model estimator; the arm's 21 counted tells are their own BH family. A tell **replicates in Gemini** if its ratio is at least 2 and BH q < 0.05; otherwise the ratio and interval are reported, with no change to the main verdicts. The other tells are reported descriptively. An OpenAI arm follows under the same rule as a separate deviation | The post's scope excludes frontier models; the founder asked for a Gemini arm on the free tier (Nov 9). Free-tier terms: Google may use prompts and outputs to improve its products and human reviewers may read them; the prompts are public HN and Stack Exchange text with no author metadata. Costs and alternatives in `frontier_arms.md` |
| Nov 9, 2026 | D15. Exploratory OpenAI arm, under D14's design, before any study request (four three-word test requests checked the parameters): `gpt-6.1-sol` at `reasoning_effort` `low` (its lowest; it accepts neither `none` nor `minimal`) and `gpt-6-luna` at `none`, through the standard Chat Completions API (not Batch: simpler, and the worst case fits the founder's US$30 cap). Same 800 prompts, verbatim, no system prompt, one sample each, seed sent as in the main arm, default sampling, `max_completion_tokens` = the registered cap plus 2,048 for reasoning (Sol) or the cap alone (Luna), the same regeneration rule (D6). A 20-prompt pilot per model on the exploratory split first; if the pilot projects more than US$25 for both models together, stop before the full run. **Hypothesis and pass rule:** as D14, for each model separately: each of P2, S1, S3 and P3 replicates in that model if its pooled confirmatory ratio against the matched human texts is at least 2 with BH q < 0.05, the model's 21 counted tells being their own family; otherwise reported with its interval, main verdicts unchanged | The founder asked for an OpenAI arm if it costs US$30 or less (Nov 9). Sol is a current mid-tier flagship; Luna, the smallest, is the only one that runs with reasoning off, like the main arm. Prices and alternatives in `frontier_arms.md` |
| Nov 9, 2026 | D16. The Gemini arm (D14) is withdrawn before its full run. `gemini-3.8-flash` on the free tier answered most requests with 503 "high demand": 11 of the 20 pilot texts in about 35 minutes, 5 of them in a final 10-minute window the founder set (20:40-20:50); 800 texts would have taken days. Founder's decision, made before any Gemini text was measured. The 11 pilot texts are kept (`data/generated/gemini.jsonl`) and not analysed. No switch to another Gemini model | A switch to `gemini-3.6-flash`, which answered, was considered and not taken: it would change the registered model after the pilot. The post's frontier coverage is OpenAI (D15) and, from Nov 15, Claude (D17, to be registered) |
| Nov 9, 2026 | D18. The registered pooled test (sections 6.1-6.5: estimator, Holm across the ten primary tells, BH across the counted secondary tells, verdict rules) is recalculated with every generator's texts, not only the three local models: equal weight per model and register, clustered by prompt. Founder's decision, after the OpenAI results were seen. Run now on five models (Qwen, Gemma, Muse, gpt-6.1-sol, gpt-6-luna) and again after the Claude arm (D17) on seven; the seven-model run becomes the post's headline. The three-model results are kept and reported beside it, unchanged, as the first registered run. No new texts, counters or thresholds; S9 (model-coded content) stays three-model only | The test is the same; only the model list grows. The post states that models were added after the first run |
| Nov 14, 2026 | D17. Exploratory Claude API arm, under D14's design, registered before any request: `claude-sonnet-5-5` with `thinking` `between_tools` (its thinking-off setting; `disabled` is rejected on this model; effort left at its default, `high`) and `claude-opus-5-5` at `output_config.effort` `low` (thinking cannot be switched off on this model; adaptive), through the Message Batches API (`llm_tells.claude_batch`). Same 800 prompts, verbatim, no system prompt, one sample each; no temperature or top-p (rejected on these models) and no seed (the API has none), so sampling is at Anthropic's defaults and unseeded; `max_tokens` = the registered cap (Sonnet) or the cap plus 2,048 for thinking (Opus); the same regeneration rule (D6), with a `refusal` stop reason also counted as a failure; no server-side fallbacks (a fallback answers with a different model); the served model ID and stop reason are stored per text. A 20-prompt pilot per model on the exploratory split first, for tokens, refusals and truncation; if it projects more than US$25 for both models together, stop before the full run. **Hypothesis and pass rule:** as D14 and D15, for each model separately: each of P2, S1, S3 and P3 replicates in that model if its pooled confirmatory ratio against the matched human texts is at least 2 with BH q < 0.05, the model's 21 counted tells being their own family; otherwise reported with its interval, main verdicts unchanged. Then the D18 pooled test on seven models. A Fable arm is added, as its own deviation, only if Sonnet and Opus disagree on any of the four tells (one replicates, the other does not). The Claude Code arm (100 texts, run with this workspace's style instructions in context) is superseded: kept and reported without a verdict, not pooled | Decided by the founder Nov 9: the Claude Code arm is a different treatment (review C3; analysis review finding 9). The two models are the current Sonnet and Opus; costs in `frontier_arms.md` |
| Nov 14, 2026 | D19. Found after the results, by a test written for the release (Nov 14): the continuation sample did not apply the 600-word upper limit that section 3.1's "same frame and rules" carries over from the main Stack Exchange sample. 27 of the 125 continuation items are over 600 words after cleaning (longest 1,270). The arm stays as run and its results as reported. Added, written down before running: the three continuation comparisons (Instruct versus base, Instruct versus human, base versus human) rerun, unchanged in estimator, seed and correction, without those 27 items (`analysis/continuation_max600/`, `llm_tells.continuation_sensitivity`), reported beside the original. No pass rule: it shows whether the arm's direction depends on the long items. The bug is fixed in `stackexchange.sample_site` for any future draw | Found by `tests/test_stackexchange.py`; reported as a deviation because it changes the sample the arm describes |

## References

Cited in full in [methodology.md](./methodology.md), reading list. Added here: Zou, G. (2004). A modified Poisson regression approach to prospective studies with binary data. *American Journal of Epidemiology* 159(7), 702-706. https://doi.org/10.1093/aje/kwh090. Hadgu, A. (1996). The discrepancy in discrepant analysis. *The Lancet* 348(9027), 592-593. Miller, W. C. (1998). Bias in discrepant analysis: when two wrongs don't make a right. *Journal of Clinical Epidemiology* 51(3), 219-231.
