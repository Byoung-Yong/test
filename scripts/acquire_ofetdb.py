from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests

FILES = {
    "combined_seed_data.xlsx": {
        "url": "https://raw.githubusercontent.com/aaronliu64/ofetdb_public/main/create_ofetdb/combined/combined_seed_data.xlsx",
        "git_blob_sha1": "dcb60dceddd5e130e26d9bbfdce589a00229726d",
    },
    "combined_measurements.xlsx": {
        "url": "https://raw.githubusercontent.com/aaronliu64/ofetdb_public/main/create_ofetdb/combined/combined_measurements.xlsx",
        "git_blob_sha1": "b54676a8ba10d916a5b5e52b433cd12de1523b69",
    },
}


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def main() -> None:
    raw_dir = Path("ofet_raw/source")
    csv_dir = Path("ofet_raw/csv")
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"source_repository": "aaronliu64/ofetdb_public", "files": {}}
    for filename, spec in FILES.items():
        response = requests.get(spec["url"], timeout=120)
        response.raise_for_status()
        data = response.content
        actual_blob = git_blob_sha1(data)
        if actual_blob != spec["git_blob_sha1"]:
            raise RuntimeError(f"Blob SHA mismatch for {filename}: {actual_blob}")

        path = raw_dir / filename
        path.write_bytes(data)
        xls = pd.ExcelFile(path)
        sheet_info = []
        for sheet in xls.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            output = csv_dir / f"{Path(filename).stem}__{safe_name(sheet)}.csv"
            frame.to_csv(output, index=False)
            sheet_info.append({
                "sheet": sheet,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "csv": str(output),
            })

        manifest["files"][filename] = {
            "url": spec["url"],
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": actual_blob,
            "sheets": sheet_info,
        }

    Path("ofet_raw/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
