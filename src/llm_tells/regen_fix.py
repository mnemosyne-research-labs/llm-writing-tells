"""Deviation D6: restore first-attempt samples that were regenerated under the old English check."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from llm_tells.generate import CONTEXT, LMS, MODELS, PARALLEL, failed, request


def main() -> None:
    """Re-request attempt 0 for regenerated or failed texts of one model and append them to a patch file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("out_path", type=Path, help="generation output, read only")
    parser.add_argument(
        "patch_path",
        type=Path,
        help="JSON-lines patch file; later records replace earlier ones by text_id",
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--before", required=True, help="ISO time when the new English rule took effect"
    )
    args = parser.parse_args()
    spec = MODELS[args.model]
    with args.out_path.open() as lines:
        records: list[dict[str, Any]] = [json.loads(line) for line in lines]
    old_rule = [
        r
        for r in records
        if r["source"] == spec.name and r["generated_at"] < args.before
    ]
    targets = [r for r in old_rule if r.get("attempt") == 1 or r.get("failed")]
    if not targets:
        print("nothing to fix")
        return
    prompts = {
        row["prompt_id"]: row
        for row in pl.read_parquet(args.prompts).iter_rows(named=True)
    }
    subprocess.run(
        [
            LMS,
            "load",
            spec.key,
            "-y",
            "-c",
            str(CONTEXT),
            "--parallel",
            str(PARALLEL),
            "--identifier",
            spec.name,
        ],
        check=True,
        capture_output=True,
    )
    with httpx.Client(timeout=900) as client, ThreadPoolExecutor(PARALLEL) as pool:
        first = list(
            pool.map(
                partial(request, client, spec, attempt=0),
                [prompts[r["prompt_id"]] for r in targets],
            )
        )
    subprocess.run([LMS, "unload", spec.name], check=True, capture_output=True)
    fixed: dict[str, dict[str, Any]] = {}
    for record, retry in zip(targets, first, strict=True):
        if not failed(retry["raw"]):
            fixed[record["text_id"]] = {
                **record,
                **retry,
                "attempt": 0,
                "failed": False,
                "truncated": retry["finish_reason"] == "length",
                "d6": "first attempt restored",
            }
        else:
            fixed[record["text_id"]] = {
                **record,
                "failed": failed(record["raw"]),
                "d6": "kept second attempt",
            }
    with args.patch_path.open("a") as patch:
        patch.writelines(json.dumps(record) + "\n" for record in fixed.values())
    restored = sum(v["d6"] == "first attempt restored" for v in fixed.values())
    print(
        f"{len(targets)} targets; {restored} first attempts restored; {sum(v['failed'] for v in fixed.values())} still failed"
    )


if __name__ == "__main__":
    main()
