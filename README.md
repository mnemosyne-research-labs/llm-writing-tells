# What gives away AI writing: data and code

The data and code behind the Mnemosyne Research Labs proof sketch [We measured what gives away AI writing](https://mnemosyneresearch.com/findings/what-gives-away-ai-writing) (MRL:2611.0002v1, November 16, 2026). Seven models answered the same 800 Hacker News comments and Stack Exchange questions that people answered between 2019 and 2022, and 23 claimed signs of AI writing ("tells") were counted in both. The plan was written and dated before any text was collected: [PREREGISTRATION.md](PREREGISTRATION.md), with nineteen dated deviations.

Code and our data are Apache 2.0 ([LICENSE](LICENSE)). The human texts are not here; see [Not included](#not-included).

## What's here

| Path | What |
| --- | --- |
| `src/llm_tells/` | All code: sampling, cleaning, generation, the tell counters, coding, analysis, the release build and the restore step |
| `data/corpus/` | The prompt tables: one row per prompt with its IDs, register, site, date, target length and split (exploratory or confirmatory); no text |
| `data/raw/*/*.ids.json` | Hacker News item IDs and Stack Exchange answer IDs, in sample order, and the exclusion counts from sampling |
| `data/generated/*.jsonl` | One record per model reply with its settings, stop reason, attempt and token usage. With the text: Qwen 3.8 27B, Gemma 4 E4B and Muse Glimmer 30B (`main.jsonl`, with first-attempt restorations in `d6_patch.jsonl`) and Llama 3.1 8B base and Instruct continuations (`continuation.jsonl`). Without the text: gpt-6.1-sol and gpt-6-luna (`openai.jsonl`), Claude Sonnet 5.5 and Opus 5.5 (`claude_api.jsonl`) |
| `data/counts/*.parquet` | One row per text, human and model, with every tell count; no text |
| `coding/` | The codebook, the coders' codes (Claude Haiku, GLM-4.7-Flash, Claude Fable and the human check) by opaque ID, and the agreement tables |
| `analysis/` | The switch rule, the power simulation, the registered three-model run (`confirmatory/`), the five- and seven-model runs (`frontier/`, `frontier_7/`), and the continuation arm without its 27 answers over 600 words (`continuation_max600/`, deviation D19) |
| `tests/` | The test suite (synthetic data only, no network); `just test` runs it with the coverage floor |
| `typings/` | Type stubs for the parts of scikit-learn and statsmodels the code uses |

## Not included

- **Human texts.** Hacker News comments belong to their authors and Stack Exchange posts are CC BY-SA 4.0 by theirs, so only IDs and our labels are released. `llm_tells.restore` downloads them again by ID.
- **38 model replies** that repeat 12 or more consecutive words of the human text they answer (mostly Llama 3.1 8B base copying the answer it was asked to continue). Their counts are in `data/counts/`; their text is withheld, listed in `data/generated/withheld.json`.
- **The text of the OpenAI and Claude replies** (3,200 replies). OpenAI's and Anthropic's terms bar using their outputs to train AI models, and an open licence on the text would let anyone do that, so only the records and counts are released. Regenerate them with your own keys (below).
- **The exploratory Claude Code arm** (97 texts written with the lab's style instructions in context, replaced by the API arm) and the 11 pilot texts of the withdrawn Gemini arm. The Claude Code arm's counts are in `data/counts/main.parquet` (source `claude`).

## Reproduce

Python 3.14 and [uv](https://docs.astral.sh/uv/). `just sync` installs the locked environment.

**The analysis, from the released counts** (about 2 minutes):

```sh
uv run python -I -m llm_tells.report data/counts/main.parquet analysis/switch_rule.json out/confirmatory \
  --continuation data/counts/continuation.parquet --recent data/counts/recent_hn.parquet --coding coding
uv run python -I -m llm_tells.frontier data/counts/frontier.parquet analysis/switch_rule.json out/frontier_7 \
  --models gpt_sol gpt_luna claude_sonnet claude_opus --main data/counts/main.parquet
```

Every table in `out/` matches `analysis/` byte for byte, except one row: the character n-gram ceiling in `discrimination.csv` needs the texts, so it is left out when they are absent.

**From the texts.** Restore the human texts (about a minute; uses about 30 of the Stack Exchange API's 300 keyless daily requests), rebuild the prompt tables keeping the released splits, and count again:

```sh
uv run python -I -m llm_tells.restore data/raw
uv run python -I -m llm_tells.corpus data/raw out/corpus --previous data/corpus/prompts.parquet
uv run python -I -m llm_tells.measure out/corpus/prompts.parquet out/main.parquet \
  --generated data/generated/main.jsonl data/generated/d6_patch.jsonl
uv run python -I -m llm_tells.measure out/corpus/recent_hn.parquet out/recent.parquet
```

Checked on November 14, 2026: all 1,209 human texts restored, and every tell count for the human texts and the released replies matches the released counts; withheld replies are skipped. The download kept a cleaned copy of each text made with the cleaning rules as they were before deviation D4; the restore cleans with the current rules. That changes the target length (`target_words`) of two prompts, and the human half of two continuation items (`cont:english:509249` and `cont:english:572253`: word, sentence and sentence-length counts). Some question titles were also edited after the download, which changes the prompt text but no count.

**Generating the replies again** needs the models: the local ones through LM Studio (`llm_tells.generate`, settings in the code), the hosted ones through their APIs with key files in a folder named by `LLM_TELLS_CREDS` (`llm_tells.generate` for OpenAI, `llm_tells.claude_batch` for Anthropic). Sampling was seeded where the API allowed it; Anthropic's API takes no seed, so its replies will not repeat exactly.

## Terms for the model replies

Every reply was written by the model named in its record (`source`, and `model` where the API reported it), not by a person, and was not edited. Mnemosyne Research Labs generated them for this study. Their factual claims were not checked and may be wrong. Qwen, Gemma and Muse Glimmer are Apache 2.0. Built with Llama: the Llama 3.1 replies are subject to the Llama 3.1 Community License and Acceptable Use Policy.

## Corrections

Found an error in the data, the code or the post? [Open an issue](https://github.com/mnemosyne-research-labs/llm-writing-tells/issues) with what you ran and what you expected. Corrections that change a number or a conclusion are published as a new version of the post, with a dated note saying what changed. Report security problems privately, as the [security policy](https://github.com/mnemosyne-research-labs/.github/blob/main/SECURITY.md) explains.

## Cite

Cite the post for the findings, and the archived release for the data and code. [doi:10.5281/zenodo.24286168](https://doi.org/10.5281/zenodo.24286168) is this version (v1.0.0). [doi:10.5281/zenodo.24286167](https://doi.org/10.5281/zenodo.24286167) always resolves to the latest version.

```bibtex
@techreport{ouyang2026tells,
  author      = {Ouyang, Flavia},
  title       = {We measured what gives away {AI} writing},
  institution = {Mnemosyne Research Labs},
  number      = {MRL:2611.0002v1},
  year        = {2026},
  month       = nov,
  url         = {https://mnemosyneresearch.com/findings/what-gives-away-ai-writing}
}
```

```bibtex
@software{ouyang2026tellsdata,
  author    = {Ouyang, Flavia},
  title     = {What gives away {AI} writing: data and code},
  publisher = {Zenodo},
  version   = {v1.0.0},
  year      = {2026},
  month     = nov,
  doi       = {10.5281/zenodo.24286168},
  url       = {https://doi.org/10.5281/zenodo.24286168}
}
```
