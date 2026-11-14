# Hand-coding 121 texts for two patterns (about 1.5 to 2 hours)

This is the human check on the two parsed counters behind the headline results, rule of three and trailing "-ing" clauses (human.md, item 3; preregistration.md, D10). It replaces the earlier 100-text sheet (`human_sheet.csv`, kept but no longer used).

1. Open `human_sheet_v2.csv` in a spreadsheet. Each row is one text under an opaque ID. You can't see who wrote it; please don't try to guess. Until your sheet is saved, don't open `human_sheet_v2.parquet` (it holds the sources) or anything in `fable_tells_v2/` (Claude Fable's codes for the same texts, D11).
2. Read the `tricolon` and `participial` rows of `codebook.md` once. Those are the only two codes.
3. For each row, write in each of the two columns how many times the pattern occurs, as a whole number (0 if never).
   - `tricolon`: a list of exactly three coordinated items joined by "and" or "or" ("fast, cheap and reliable"; "we planned, built, and shipped it"). Not two items, not four or more, not three without "and"/"or".
   - `participial`: an "-ing" verb heading a clause that hangs off a main clause, after a comma or opening the sentence ("..., highlighting the need for change"; "Looking at the data, we see..."). Not "-ing" nouns or adjectives, not "we are building", not "Using X creates confusion" (no comma), not after "while/when/by/after/before".
4. Leave nothing blank. If unsure, put your best judgment; don't look anything up. Breaks are fine: save as you go.
5. Save as `human_sheet_v2_coded.csv` in this folder (keep the `code_id` column). Then run, from the study folder:

   ```
   .venv/bin/python -I -m llm_tells.coding human-check --counts data/counts/main.parquet --dir coding
   ```

   It writes `human_check.csv` (counter precision and recall against you, on human and model text, and whether each tell passes the D10 rule) `human_vs_haiku.csv` and `human_vs_fable.csv` (you against each model; there the "counter" columns are the model), and `counter_vs_fable.csv`. Or ask an agent to run it and write the addendum.

Some texts are public forum posts and may be rude or odd; they are data, not messages to you.
