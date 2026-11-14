"""Re-download the human texts of a release by ID, in the format the corpus step reads."""

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from llm_tells.hn import CONCURRENCY, check
from llm_tells.stackexchange import (
    EXCLUDE,
    FIELDS,
    StackExchangeClient,
    attach_questions,
)

HN_END_TIMES = {"hn_2019_2022": 1669766400, "hn_2025_2026": 1793491200}
PAGE = 100


def hn_ids(records: list[dict[str, Any]]) -> list[int]:
    """Item IDs of HN reply records, in file order.

    Args:
        records: Records from a `replies.jsonl`

    Returns:
        The IDs
    """
    return [record["id"] for record in records]


def se_ids(records: list[dict[str, Any]]) -> list[list[str | int]]:
    """Site and answer ID of Stack Exchange records, in file order.

    Args:
        records: Records from an answers file

    Returns:
        `[site, answer_id]` pairs
    """
    return [[record["site"], record["answer_id"]] for record in records]


async def fetch_hn(
    ids: list[int], end_time: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Fetch HN replies by ID with the sampler's own checks.

    Args:
        ids: Reply item IDs, in the order to write them
        end_time: The sample's creation-time cutoff

    Returns:
        The records in ID order, and the IDs no longer eligible (deleted or changed)
    """
    limit = asyncio.Semaphore(CONCURRENCY)

    async def one(client: httpx.AsyncClient, item_id: int) -> dict[str, Any] | None:
        async with limit:
            reply, _ = await check(client, item_id, end_time)
        return None if reply is None else asdict(reply)

    async with httpx.AsyncClient(timeout=60) as client:
        found = await asyncio.gather(*(one(client, item_id) for item_id in ids))
    missing = [
        item_id for item_id, record in zip(ids, found, strict=True) if record is None
    ]
    return [record for record in found if record is not None], missing


def fetch_se(pairs: list[list[str | int]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Fetch Stack Exchange answers and their questions by ID.

    Args:
        pairs: `[site, answer_id]` pairs, in the order to write them

    Returns:
        The records in input order, and the answer IDs not found
    """
    by_site: defaultdict[str, list[int]] = defaultdict(list)
    for site, answer_id in pairs:
        by_site[str(site)].append(int(answer_id))
    found: dict[tuple[str, int], dict[str, Any]] = {}
    with httpx.Client(timeout=60) as client:
        api = StackExchangeClient(client)
        filter_id = api.get(
            "/filters/create",
            include=";".join(FIELDS),
            exclude=";".join(EXCLUDE),
            base="default",
            unsafe="false",
        )["items"][0]["filter"]
        for site, ids in by_site.items():
            answers = [
                answer
                for start in range(0, len(ids), PAGE)
                for answer in api.get(
                    f"/answers/{';'.join(map(str, ids[start : start + PAGE]))}",
                    site=site,
                    pagesize=PAGE,
                    filter=filter_id,
                )["items"]
            ]
            found |= {
                (site, record.answer_id): asdict(record)
                for record in attach_questions(api, site, answers, filter_id)
            }
    order = [(str(site), int(answer_id)) for site, answer_id in pairs]
    return (
        [found[key] for key in order if key in found],
        [key[1] for key in order if key not in found],
    )


def restore(raw_dir: Path) -> dict[str, list[int]]:
    """Write every `<name>.jsonl` next to its `<name>.ids.json` under `raw_dir`.

    Args:
        raw_dir: The release's `data/raw` folder

    Returns:
        IDs that could not be restored, per file
    """
    missing: dict[str, list[int]] = {}
    for ids_path in sorted(raw_dir.glob("*/*.ids.json")):
        ids = json.loads(ids_path.read_text())
        folder = ids_path.parent.name
        if folder in HN_END_TIMES:
            records, gone = asyncio.run(fetch_hn(ids, HN_END_TIMES[folder]))
        else:
            records, gone = fetch_se(ids)
        target = ids_path.with_name(ids_path.name.removesuffix(".ids.json") + ".jsonl")
        target.write_text("".join(json.dumps(record) + "\n" for record in records))
        missing[str(target.relative_to(raw_dir))] = gone
        print(
            f"{target.relative_to(raw_dir)}: {len(records)} restored, {len(gone)} missing",
            flush=True,
        )
    return missing


def main() -> None:
    """Restore the human texts of a release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="the release's data/raw folder")
    args = parser.parse_args()
    missing = restore(args.raw_dir)
    (args.raw_dir / "missing.json").write_text(json.dumps(missing, indent=2))


if __name__ == "__main__":
    main()
