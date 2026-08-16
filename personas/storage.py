"""One JSON file per record, so a spot preemption loses at most one call."""
import hashlib
import json
import os
from pathlib import Path


def shard_key(arm_id: str, rung: str, condition: str, item_id: str) -> str:
    return f"{arm_id}|{rung}|{condition}|{item_id}"


def _filename(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16] + ".json"


def completed_keys(output_prefix: str) -> set[str]:
    root = Path(output_prefix)
    if not root.exists():
        return set()
    keys = set()
    for path in root.glob("*.json"):
        try:
            keys.add(json.loads(path.read_text())["key"])
        except (json.JSONDecodeError, KeyError):
            path.unlink()  # torn write from a preemption; drop it
    return keys


def write_record(output_prefix: str, record: dict) -> None:
    root = Path(output_prefix)
    root.mkdir(parents=True, exist_ok=True)
    target = root / _filename(record["key"])
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, target)  # atomic, so readers never see a partial file
