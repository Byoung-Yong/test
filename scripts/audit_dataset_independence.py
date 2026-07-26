from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("ofet_hidden_validation")
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def rounded_counter(series: pd.Series, digits: int = 12) -> Counter[float]:
    return Counter(np.round(pd.to_numeric(series, errors="coerce").dropna().to_numpy(float), digits))


def main() -> None:
    ofet = pd.read_csv(DATA / "ofetdb_mobility_harmonized.csv")
    external = pd.read_csv(DATA / "p3ht_218_harmonized.csv")
    ofet_p3ht = ofet[ofet["family"] == "P3HT"].copy()

    target_equal = rounded_counter(ofet_p3ht["log10_mobility"]) == rounded_counter(external["log10_mobility"])
    mobility_equal = rounded_counter(ofet_p3ht["mobility"]) == rounded_counter(external["mobility"])
    source_overlap = sorted(set(ofet_p3ht["source"].astype(str)) & set(external["source"].astype(str)))

    comparable = [
        "mn_kda", "mw_kda", "dispersity", "regioregularity_pct", "concentration_mg_ml",
        "solvent", "electrode_material", "device_configuration", "channel_length_um",
        "channel_width_um", "substrate_pretreat", "deposition_type", "spin_rate_rpm",
        "spin_time_s", "anneal_temperature_c", "anneal_time", "mobility_regime",
        "measurement_environment", "vds_v",
    ]
    feature_overlap: dict[str, object] = {}
    for col in comparable:
        a = ofet_p3ht[col]
        b = external[col]
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
            aset = set(np.round(pd.to_numeric(a, errors="coerce").dropna(), 8))
            bset = set(np.round(pd.to_numeric(b, errors="coerce").dropna(), 8))
        else:
            aset = set(a.dropna().astype(str).str.strip().str.lower()) - {"__missing__"}
            bset = set(b.dropna().astype(str).str.strip().str.lower()) - {"__missing__"}
        union = aset | bset
        feature_overlap[col] = {
            "ofet_unique": len(aset),
            "external_unique": len(bset),
            "intersection": len(aset & bset),
            "jaccard": float(len(aset & bset) / len(union)) if union else None,
        }

    same_underlying_cohort = bool(
        len(ofet_p3ht) == len(external)
        and target_equal
        and mobility_equal
    )

    audit = {
        "ofetdb_p3ht_rows": int(len(ofet_p3ht)),
        "external_p3ht_rows": int(len(external)),
        "ofetdb_p3ht_sources": int(ofet_p3ht["source"].nunique()),
        "external_p3ht_sources": int(external["source"].nunique()),
        "source_overlap_count": len(source_overlap),
        "source_overlap": source_overlap,
        "log10_mobility_multiset_identical": bool(target_equal),
        "mobility_multiset_identical": bool(mobility_equal),
        "same_underlying_cohort": same_underlying_cohort,
        "independent_dataset_test_valid": not same_underlying_cohort,
        "feature_value_overlap": feature_overlap,
        "adjudication": (
            "The two files represent the same 218-record P3HT cohort; the external result is a family-hidden replay, not an independent dataset-hidden validation."
            if same_underlying_cohort
            else "No exact cohort identity was detected from row count and target multisets."
        ),
    }
    (RESULTS / "dataset_independence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    gate = json.loads((RESULTS / "gate.json").read_text(encoding="utf-8"))
    gate["dataset_independence_confirmed"] = bool(not same_underlying_cohort)
    gate["dataset_and_family_hidden_pass_adjudicated"] = bool(
        gate.get("dataset_and_family_hidden_pass", False) and not same_underlying_cohort
    )
    gate["universal_mobility_gate_pass_adjudicated"] = bool(
        gate.get("family_rotation_all_pass", False)
        and gate["dataset_and_family_hidden_pass_adjudicated"]
    )
    gate["dataset_adjudication"] = audit["adjudication"]
    (RESULTS / "gate_adjudicated.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
