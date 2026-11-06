"""Build the shared experiment index from completed, failed, and interrupted runs."""

import csv
import json
from pathlib import Path

from .artifacts import read_json
from .contracts import require
from .provenance import file_record

COLUMNS = ("run", "kind", "status", "started_at", "revision", "source_sha256", "config_sha256",
           "method", "data_kind", "dataset_id", "manifest_sha256", "split", "elapsed_seconds",
           "output_sha256", "comparison_results", "report")


def export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(root.is_dir(), "experiment root must be an existing directory")
    require(out.suffix.lower() == ".csv", "experiment index must be a CSV file")
    if out.exists():
        with out.open(encoding="utf-8", newline="") as handle:
            require(next(csv.reader(handle), None) == list(COLUMNS),
                    "refusing to overwrite a file that is not a P4 experiment index")
    rows = []
    for path in sorted(root.rglob("run.json")):
        record = read_json(path)
        report_path = record.get("report", {}).get("path")
        report = read_json(report_path) if report_path and record["status"] == "complete" else {}
        if report:
            require(file_record(report_path) == record["report"], "experiment report changed: " + report_path)
        rows.append({"run": str(path.parent.relative_to(root)), "kind": record["kind"], "status": record["status"],
                     "started_at": record.get("started_at", ""), "revision": record.get("source", {}).get("revision", ""),
                     "source_sha256": record.get("source", {}).get("sha256", ""),
                     "config_sha256": record.get("config_sha256", ""), "method": record.get("method", ""),
                     "data_kind": record.get("data_kind", record.get("evidence_kind", "")),
                     "dataset_id": record.get("dataset_id", ""), "manifest_sha256": record.get("manifest", {}).get("sha256", ""),
                     "split": record.get("split", ""), "elapsed_seconds": record.get("elapsed_seconds", ""),
                     "output_sha256": record.get("output", {}).get("sha256", ""),
                     "comparison_results": json.dumps(report["summary"], sort_keys=True) if "summary" in report else "",
                     "report": report_path or ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
