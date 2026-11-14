"""Build the public release: code, IDs, counts, model replies and analysis, with no human text."""

import argparse
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl

from llm_tells.restore import hn_ids, se_ids

STUDY = Path(__file__).resolve().parents[2]
QUOTE_WORDS = 12
TEMPLATE_SPREAD = 20
TEXT_COLUMNS = frozenset(
    {
        "prompt",
        "human_raw",
        "base_prompt",
        "raw",
        "clean_text",
        "text",
        "flagged_phrases",
        "instances",
        "note",
    }
)
CODE_FILES = (".python-version", "pyproject.toml", "uv.lock")
DOC_FILES = {
    "preregistration.md": "PREREGISTRATION.md",
    "coding/codebook.md": "coding/codebook.md",
    "coding/human_instructions.md": "coding/human_instructions.md",
    "coding/adjudication_instructions.md": "coding/adjudication_instructions.md",
}
GENERATED = (
    "main.jsonl",
    "d6_patch.jsonl",
    "continuation.jsonl",
    "openai.jsonl",
    "claude_api.jsonl",
)
# Hosted providers' terms bar using their outputs to train AI models, which an open
# licence would allow, so these replies are released as counts only.
HOSTED_TERMS = (
    "hosted-model reply, released as counts only: the provider's terms bar using it to "
    "train AI models; regenerate it with your own key"
)
HOSTED_GENERATED = frozenset({"openai.jsonl", "claude_api.jsonl"})
GENERATED_TEXT_FIELDS = ("raw", "uncut")
COPIED_AS_IS = (
    "data/generated/runs.json",
    "data/model_hashes.txt",
    "data/hn_queries.txt",
)
TABLE_GLOBS = (
    "data/corpus/*.parquet",
    "data/counts/*.parquet",
    "coding/*.csv",
    "coding/*.parquet",
)
ANALYSIS_SUFFIXES = frozenset({".csv", ".json", ".md", ".log"})
HUMAN_SOURCES = (
    ("data/corpus/prompts.parquet", ("prompt", "human_raw")),
    ("data/corpus/continuation.parquet", ("prompt", "human_raw")),
    ("data/corpus/recent_hn.parquet", ("prompt", "human_raw")),
)
WORD_RE = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    """Lowercase word tokens used for quote detection.

    Args:
        text: Any text

    Returns:
        The tokens
    """
    return WORD_RE.findall(text.lower())


def ngrams(text: str) -> Iterator[tuple[str, ...]]:
    """Every run of `QUOTE_WORDS` consecutive words.

    Args:
        text: Any text

    Yields:
        Word tuples
    """
    tokens = words(text)
    return (
        tuple(tokens[i : i + QUOTE_WORDS]) for i in range(len(tokens) - QUOTE_WORDS + 1)
    )


def human_ngrams() -> frozenset[tuple[str, ...]]:
    """All word runs of the human texts and the prompts built from them.

    Runs found in many texts are the prompt template, not human writing, and are
    left out.

    Returns:
        The set of runs a released file must not contain
    """
    texts = (
        text
        for path, columns in HUMAN_SOURCES
        for column in columns
        for text in pl.read_parquet(STUDY / path)[column].drop_nulls()
    )
    spread = Counter(gram for text in texts for gram in set(ngrams(text)))
    return frozenset(gram for gram, count in spread.items() if count < TEMPLATE_SPREAD)


def quotes_human(text: str | None, banned: frozenset[tuple[str, ...]]) -> bool:
    """Whether a text repeats any run of human text.

    Args:
        text: Text to check, or None
        banned: Human word runs

    Returns:
        True if any run matches
    """
    return text is not None and any(gram in banned for gram in ngrams(text))


def strip_table(source: Path, target: Path) -> None:
    """Copy a CSV or parquet table without its text columns.

    Args:
        source: Study table
        target: Release path (same format)
    """
    is_csv = source.suffix == ".csv"
    frame = (
        pl.read_csv(source, infer_schema_length=0)
        if is_csv
        else pl.read_parquet(source)
    )
    frame = frame.drop(TEXT_COLUMNS & set(frame.columns))
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_csv:
        frame.write_csv(target)
    else:
        frame.write_parquet(target)


def release_generated(
    source: Path,
    target: Path,
    banned: frozenset[tuple[str, ...]],
    withhold_all: str | None = None,
) -> list[str]:
    """Copy model replies, withholding the text of any that quotes human text.

    Args:
        source: Study JSON-lines file
        target: Release JSON-lines file
        banned: Human word runs
        withhold_all: If given, every reply's text is withheld, with this reason

    Returns:
        IDs of the texts withheld because they quote human text
    """
    withheld: list[str] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as lines, target.open("w") as out:
        for line in lines:
            record = json.loads(line)
            if withhold_all is not None:
                record |= {
                    field: None for field in GENERATED_TEXT_FIELDS if field in record
                }
                record["withheld"] = withhold_all
            elif any(
                quotes_human(record.get(field), banned)
                for field in GENERATED_TEXT_FIELDS
            ):
                withheld.append(record["text_id"])
                record |= {
                    field: None for field in GENERATED_TEXT_FIELDS if field in record
                }
                record["withheld"] = f"quotes {QUOTE_WORDS} or more words of human text"
            out.write(json.dumps(record) + "\n")
    return withheld


def released_strings(path: Path) -> Iterable[str]:
    """Every string value in a released file, for the final leak scan.

    Args:
        path: Released file

    Returns:
        The strings (whole text for non-table files)
    """
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
        return (
            value
            for column, dtype in frame.schema.items()
            if dtype == pl.String
            for value in frame[column].drop_nulls()
        )
    return (path.read_text(errors="ignore"),)


def leaks(root: Path, banned: frozenset[tuple[str, ...]]) -> list[str]:
    """Released files that still contain a run of human text.

    Args:
        root: Release root
        banned: Human word runs

    Returns:
        Relative paths of offending files
    """
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix
        in {".parquet", ".csv", ".json", ".jsonl", ".md", ".txt", ".log"}
        and any(quotes_human(text, banned) for text in released_strings(path))
    )


def write_ids(source: Path, target: Path) -> None:
    """Write the IDs of a downloaded human-text file, so `llm_tells.restore` can refetch it.

    Args:
        source: A `data/raw` JSON-lines file
        target: Its release path; `<name>.ids.json` is written beside it
    """
    with source.open() as lines:
        records = [json.loads(line) for line in lines]
    ids = se_ids(records) if "answer_id" in records[0] else hn_ids(records)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(f"{target.stem}.ids.json").write_text(json.dumps(ids))


def copy(source: Path, target: Path) -> None:
    """Copy one file, creating its folder.

    Args:
        source: File to copy
        target: Destination
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build(out: Path) -> dict[str, list[str]]:
    """Write the release into `out`, replacing earlier release files but not `.git`.

    Args:
        out: Release repository folder

    Returns:
        Withheld reply IDs per generated file

    Raises:
        RuntimeError: If any released file still contains human text
    """
    for name in ("src", "tests", "data", "coding", "analysis"):
        shutil.rmtree(out / name, ignore_errors=True)
    for path in (STUDY / "src/llm_tells").glob("*.py"):
        copy(path, out / "src/llm_tells" / path.name)
    for path in (STUDY / "tests").glob("*.py"):
        copy(path, out / "tests" / path.name)
    for name in CODE_FILES:
        copy(STUDY / name, out / name)
    for source, target in DOC_FILES.items():
        copy(STUDY / source, out / target)
    for name in COPIED_AS_IS:
        copy(STUDY / name, out / name)
    for path in (STUDY / "data/raw").rglob("*.json"):
        copy(path, out / path.relative_to(STUDY))
    for path in (STUDY / "data/raw").rglob("*.jsonl"):
        write_ids(path, out / path.relative_to(STUDY))
    for path in (STUDY / "analysis").rglob("*"):
        if path.suffix in ANALYSIS_SUFFIXES:
            copy(path, out / path.relative_to(STUDY))
    for pattern in TABLE_GLOBS:
        for path in STUDY.glob(pattern):
            strip_table(path, out / path.relative_to(STUDY))
    for path in (STUDY / "coding").glob("*/*.out.json"):
        copy(path, out / path.relative_to(STUDY))
    banned = human_ngrams()
    withheld = {
        name: release_generated(
            STUDY / "data/generated" / name,
            out / "data/generated" / name,
            banned,
            HOSTED_TERMS if name in HOSTED_GENERATED else None,
        )
        for name in GENERATED
    }
    if found := leaks(out, banned):
        raise RuntimeError(f"Human text found in released files: {', '.join(found)}")
    return withheld


def main() -> None:
    """Build the release and report the withheld replies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="release repository folder")
    args = parser.parse_args()
    withheld = build(args.out)
    (args.out / "data/generated/withheld.json").write_text(
        json.dumps(withheld, indent=2)
    )
    for name, ids in withheld.items():
        print(f"{name}: {len(ids)} replies withheld")


if __name__ == "__main__":
    main()
