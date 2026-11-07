"""Read-only B01 image inspection. No model API calls or source image writes."""

import hashlib
import io
import json
import platform
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "A03_候选图片来源清单.jsonl"
RECORDS_PATH = ROOT / "B01_图片检查清单.jsonl"
SUMMARY_PATH = ROOT / "B01_图片检查汇总.json"
REFERENCES = (
    ("REF-MAIN", "reference_main", "E:/cstudy/AIGC/flux/flux/refer/IMG_20261001_164239.jpg"),
    ("REF-SUPPLEMENT", "reference_supplement", "E:/cstudy/AIGC/flux/flux/refer/IMG_20261001_164251.jpg"),
)


def inspection_inputs():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("A03 来源清单为空。")
    identifiers = [row["record_id"] for row in rows]
    paths = [row["source_path"] for row in rows]
    if len(set(identifiers)) != len(identifiers) or len(set(paths)) != len(paths):
        raise ValueError("A03 来源清单存在重复编号或路径。")
    inputs = [
        {"inspection_id": row["record_id"], "role": "candidate",
         "source_path": row["source_path"], "batch": row["batch"]}
        for row in rows
    ]
    inputs.extend({"inspection_id": identifier, "role": role, "source_path": path}
                  for identifier, role, path in REFERENCES)
    return inputs


def inspect_image(item):
    record = dict(item)
    record.update({"status": "failed", "path_valid": False, "decode_ok": False})
    path = Path(item["source_path"])
    if not path.is_absolute() or not path.is_file():
        record["error"] = "路径不是现存的绝对文件路径。"
        return record
    record["path_valid"] = True
    try:
        # Hash and decode the same byte snapshot, rather than opening a changing path twice.
        raw = path.read_bytes()
        record.update({"file_size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                orientation = image.getexif().get(274, 1)
                width, height = image.size
                frame_count = getattr(image, "n_frames", 1)
                record.update({
                    "format": image.format, "width": width, "height": height,
                    "mode": image.mode, "frame_count": frame_count,
                    "exif_orientation": orientation,
                    "display_width": height if orientation in (5, 6, 7, 8) else width,
                    "display_height": width if orientation in (5, 6, 7, 8) else height,
                })
                for frame in range(frame_count):
                    image.seek(frame)
                    image.load()
        record.update({"status": "passed", "decode_ok": True})
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return record


def summarize(records):
    candidates = [row for row in records if row["role"] == "candidate"]
    references = [row for row in records if row["role"] != "candidate"]
    groups = defaultdict(list)
    for row in candidates:
        if row.get("sha256"):
            groups[row["sha256"]].append(row["inspection_id"])
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "pillow_version": pillow_version,
        "source_manifest": SOURCE.name,
        "source_manifest_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate_count": len(candidates), "reference_count": len(references),
        "passed_count": sum(row["status"] == "passed" for row in records),
        "failed_count": sum(row["status"] != "passed" for row in records),
        "batch_counts": dict(Counter(row["batch"] for row in candidates)),
        "candidate_dimensions": dict(Counter(
            f"{row['width']}x{row['height']}" for row in candidates if row["decode_ok"])),
        "reference_dimensions": dict(Counter(
            f"{row['width']}x{row['height']}" for row in references if row["decode_ok"])),
        "format_counts": dict(Counter(row["format"] for row in records if row["decode_ok"])),
        "candidate_byte_unique_count": len(groups),
        "candidate_byte_duplicate_groups": [ids for ids in groups.values() if len(ids) > 1],
        "failed_ids": [row["inspection_id"] for row in records if row["status"] != "passed"],
        "scope": "file_integrity_only; no quality judgment, exclusion, split or API calls",
    }


def write_report(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    records = [inspect_image(item) for item in inspection_inputs()]
    summary = summarize(records)
    write_report(RECORDS_PATH, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    write_report(SUMMARY_PATH, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
