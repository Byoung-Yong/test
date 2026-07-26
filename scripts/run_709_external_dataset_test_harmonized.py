from __future__ import annotations

from typing import Any

import numpy as np

import run_709_external_dataset_test as validation

_original_best_category = validation.best_category
_original_choose_model = validation.choose_model


def norm(value: Any) -> str:
    return validation.normalize(value)


def index_of(categories: np.ndarray, label: str):
    values = [str(x) for x in categories]
    return values.index(label) if label in values else None


def explicit_label(field: str, value: Any) -> str | None:
    v = norm(value)
    if not v or v in {"missing", "nan"}:
        if field in {"Source Electrodes Category", "Drain Electrodes Category"}:
            return "Not specified"
        return None

    if field == "Fabrication Category":
        if any(token in v for token in ["spin", "blade", "dip", "drop", "solution", "coat", "print", "inkjet", "cast"]):
            return "Solution Processing Techniques"
        if any(token in v for token in ["vacuum", "sublim", "thermal evapor", "evaporation"]):
            return "Vacuum Techniques"
        if any(token in v for token in ["physical", "vapor", "vapour", "deposition"]):
            return "Physical Deposition Techniques"
        return "Miscellaneous Techniques"

    if field in {"Source Electrodes Category", "Drain Electrodes Category"}:
        replacements = [
            (["au", "gold"], "Gold"), (["ag", "silver"], "Silver"),
            (["al", "aluminum", "aluminium"], "Aluminum"), (["pd", "palladium"], "Palladium"),
            (["cr au", "chromium gold"], "Chromium/Gold"), (["ti au", "titanium gold"], "Titanium/Gold"),
            (["ni au", "nickel gold"], "Nickel/Gold"), (["graphite"], "Graphite"),
            (["carbon nanotube", "cnt"], "CNT/Gold"),
        ]
        for aliases, label in replacements:
            if any(alias == v or alias in v for alias in aliases):
                return label
        return "Not specified"

    if field == "Gate Electrode Category":
        if "n doped" in v or "n type silicon" in v: return "n-doped Silicon"
        if "p doped" in v or "p type silicon" in v: return "p-doped Silicon"
        if "doped" in v and ("si" in v or "silicon" in v): return "Doped Si"
        if "silicon" in v or v == "si": return "Pure Silicon"
        if "indium tin" in v or "ito" in v: return "Indium Tin Oxide"
        if "gold" in v or v == "au": return "Gold"
        if "silver" in v or v == "ag": return "Silver"
        if "aluminum" in v or "aluminium" in v or v == "al": return "Aluminum"
        if any(x in v for x in ["graphene", "carbon", "cnt"]): return "Carbon-based Materials"
        return "Other Metals"

    if field == "Dielectric Layer Category":
        if "sio2" in v or "silicon dioxide" in v: return "Pure SiO2"
        if "al2o3" in v or "aluminum oxide" in v or "aluminium oxide" in v: return "Pure Al2O3"
        if any(x in v for x in ["pmma", "pvp", "cytop", "parylene", "polymer", "polystyrene", "ps"]): return "Single Polymer"
        if any(sep in v for sep in ["/", "+"]) or ("organic" in v and "inorganic" in v): return "Hybrid Organic/Inorganic Material"
        if any(x in v for x in ["sam", "ots", "hmds", "surface treatment"]): return "Surface Treatment Material"
        if any(x in v for x in ["oxide", "hfo", "zro", "tio"]): return "Other Inorganic Material"
        return "Special/Unique Material"

    if field == "Device Geometries Category":
        compact = v.replace(" ", "").upper()
        for label in ["BGBC", "BGTC", "TGBC", "TGTC"]:
            if label in compact: return label
        if "bottom gate" in v and "bottom contact" in v: return "BGBC"
        if "bottom gate" in v and "top contact" in v: return "BGTC"
        if "top gate" in v and "bottom contact" in v: return "TGBC"
        if "top gate" in v and "top contact" in v: return "TGTC"
        return None

    if field == "Conduction Type Category":
        if "ambipolar" in v: return "Ambipolar"
        if any(x in v for x in ["electron", "n type", "n channel"]): return "n-type"
        if any(x in v for x in ["hole", "p type", "p channel"]): return "p-type"
        return None

    if field == "Test Atmosphere Category":
        if "air" in v or "ambient" in v: return "Air Environment"
        if any(x in v for x in ["nitrogen", "n2", "argon", "inert", "glovebox"]): return "Inert Gas Environment"
        if "vacuum" in v: return "Vacuum Environment"
        return None
    return None


def harmonized_best_category(field: str, value: Any, categories: np.ndarray):
    label = explicit_label(field, value)
    if label is not None:
        idx = index_of(categories, label)
        if idx is not None:
            return idx, 1.0, label
    return _original_best_category(field, value, categories)


def choose_model_with_normalized_groups(X, y, groups):
    normalized = np.asarray(["__MISSING_SOURCE__" if x is None or str(x) == "nan" else str(x) for x in groups], dtype=str)
    return _original_choose_model(X, y, normalized)


validation.best_category = harmonized_best_category
validation.choose_model = choose_model_with_normalized_groups

if __name__ == "__main__":
    validation.main()
