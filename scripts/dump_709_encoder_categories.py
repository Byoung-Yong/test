from __future__ import annotations

import json
import pickle
from pathlib import Path

ROOT = Path("ofet_hidden_validation")
SOURCE = ROOT / "source_709"
RESULTS = ROOT / "results"
COLUMNS = [
    "major_category",
    "Fabrication Category",
    "Source Electrodes Category",
    "Drain Electrodes Category",
    "Gate Electrode Category",
    "Dielectric Layer Category",
    "Device Geometries Category",
    "Conduction Type Category",
    "Test Atmosphere Category",
    "publication_year",
]

with open(SOURCE / "enc.pkl", "rb") as handle:
    encoder = pickle.load(handle)

payload = {
    name: [str(value) for value in values]
    for name, values in zip(COLUMNS, encoder.categories_)
}
(RESULTS / "external_709_encoder_categories.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
