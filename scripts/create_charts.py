#!/usr/bin/env python3
"""Create the five verified Phase 5 chart files and a checksum manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skillscope.charts import EXPECTED_CHARTS, create_charts, load_verified_results  # noqa: E402
from skillscope.source import sha256_file  # noqa: E402


def main() -> int:
    analysis_dir = ROOT / "reports" / "generated" / "analysis"
    output_dir = ROOT / "reports" / "generated" / "charts"
    try:
        results = load_verified_results(
            analysis_dir,
            ROOT / "reports" / "generated" / "analysis_summary.json",
            ROOT / "data" / "processed" / "jobs.parquet",
            ROOT / "data" / "processed" / "job_skills.parquet",
        )
        outputs = create_charts(results, output_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    manifest = {
        "analysis_inputs": {
            "jobs_sha256": sha256_file(ROOT / "data" / "processed" / "jobs.parquet"),
            "job_skills_sha256": sha256_file(
                ROOT / "data" / "processed" / "job_skills.parquet"
            ),
        },
        "charts": {
            name: {
                path.suffix.lstrip("."): {
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            }
            for name, paths in outputs.items()
        },
    }
    if tuple(manifest["charts"]) != EXPECTED_CHARTS:
        print("ERROR: chart set does not match the Phase 5 contract", file=sys.stderr)
        return 1
    manifest_path = output_dir / "chart_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: wrote {len(outputs)} verified charts as PNG and SVG")
    for name, paths in outputs.items():
        print(f"- {name}: {', '.join(path.name for path in paths)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
