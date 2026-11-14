# Reviewing 62 items for the two patterns (about 1 to 1.5 hours)

This is the second pass on your hand-coding (preregistration.md, D13). Each item is one text and one pattern where your first count differed from the counter's or from Claude Fable's, or where all three agreed (a few, drawn at random, to check for mistakes all three share). The sheet doesn't say which kind an item is, or whose count is whose.

1. Open `adjudication_sheet.csv` in a spreadsheet (a Google Sheet is fine, as before). Until you have saved your review, don't open `adjudication_sheet.parquet` (it holds who said what), your first-pass sheet or anything in `fable_tells_v2/`.
2. Reread the `tricolon` or `participial` definition in `codebook.md`, and `human_coder_notes.md`, as you did for the first pass. Each row is one pattern only, named in `pattern`.
3. For each row:
   - `options`: the counts given so far, sorted, with no names. One of them may be right, or none.
   - `flagged_phrases`: what the software counter matched, separated by `|`. It is often wrong (on human text, the "-ing" counter was right about one time in five), and it misses things. Use it to find candidates, not as an answer.
   - `final`: the count you now think is correct, as a whole number. It doesn't have to be one of the options.
   - `instances`: quote each instance you count, separated by `|` (a few words each is enough to find it). Leave it blank if `final` is 0.
   - `note`: optional, for judgment calls.
4. Leave no `final` blank. If unsure, give your best judgment; don't look anything up.
5. Save as `adjudication_sheet_reviewed.csv` in this folder (keep the `item_id` column). Then run, from the study folder:

   ```
   .venv/bin/python -I -m llm_tells.coding adjudication-check --counts data/counts/main.parquet --dir coding
   ```

   It writes `human_sheet_v2_adjudicated.csv` (your codes after the review), `human_check_adjudicated.csv` and `human_check_sensitivity_adjudicated.csv` (the D10 scores on them) and `adjudication_summary.csv` (how many counts changed, and toward whom). Or ask an agent to run it and write it up.

Some texts are public forum posts and may be rude or odd; they are data, not messages to you.
