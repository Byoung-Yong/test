from __future__ import annotations

import hashlib
import json
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 20260726
ROOT = Path("ofet_hidden_validation")
SOURCE = ROOT / "source_709"
RESULTS = ROOT / "results"
OFET = ROOT / "data" / "ofetdb_mobility_harmonized.csv"

FILES = {
    "features_cat_onehot.npy": {
        "url": "https://raw.githubusercontent.com/YajingSun-Group/ofet_agent/main/1_ML_models/data/features_cat_onehot.npy",
        "git_blob_sha1": "d7b9a9c36fc8a1a98185f56b9e4a135ca82ad9c8",
    },
    "labels.npy": {
        "url": "https://raw.githubusercontent.com/YajingSun-Group/ofet_agent/main/1_ML_models/data/labels.npy",
        "git_blob_sha1": "d0b0967b8f8ebc12b09b83aaf78680a15ab77254",
    },
    "enc.pkl": {
        "url": "https://raw.githubusercontent.com/YajingSun-Group/ofet_agent/main/1_ML_models/enc.pkl",
        "git_blob_sha1": "6dd8db9a27faca6d73ee2c351a6fb016898155d3",
    },
}

ENCODER_COLUMNS = [
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
SHARED_COLUMNS = ENCODER_COLUMNS[1:9]
OFET_MAPPING = {
    "Fabrication Category": "deposition_type",
    "Source Electrodes Category": "electrode_material",
    "Drain Electrodes Category": "electrode_material",
    "Gate Electrode Category": "gate_material",
    "Dielectric Layer Category": "dielectric_material",
    "Device Geometries Category": "device_configuration",
    "Conduction Type Category": "carrier_type",
    "Test Atmosphere Category": "measurement_environment",
}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def normalize(value: Any) -> str:
    s = str(value).strip().lower()
    s = s.replace("−", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(value: Any) -> set[str]:
    return set(normalize(value).split())


def domain_aliases(field: str, value: str) -> list[str]:
    v = normalize(value)
    aliases = [v]
    replacements = {
        "au": ["gold"], "gold": ["au"],
        "ag": ["silver"], "silver": ["ag"],
        "al": ["aluminum", "aluminium"],
        "sio2": ["silicon dioxide", "sio 2"],
        "p3ht": ["poly 3 hexylthiophene"],
        "bg tc": ["bottom gate top contact", "bottom gate top electrode"],
        "bg bc": ["bottom gate bottom contact", "bottom gate bottom electrode"],
        "tg bc": ["top gate bottom contact", "top gate bottom electrode"],
        "tg tc": ["top gate top contact", "top gate top electrode"],
        "hole": ["p type", "p-type", "p channel"],
        "electron": ["n type", "n-type", "n channel"],
        "nitrogen": ["n2", "inert"],
        "spin": ["spin coating", "spin-coating", "spin coat"],
        "blade": ["blade coating", "blade-coating"],
        "dip": ["dip coating", "dip-coating"],
    }
    for key, vals in replacements.items():
        if key in v:
            aliases.extend(normalize(x) for x in vals)
    return list(dict.fromkeys(aliases))


def best_category(field: str, value: Any, categories: np.ndarray) -> tuple[int | None, float, str | None]:
    if pd.isna(value) or normalize(value) in {"", "missing", "nan"}:
        return None, 0.0, None
    normalized_categories = [normalize(x) for x in categories]
    aliases = domain_aliases(field, str(value))
    for alias in aliases:
        if alias in normalized_categories:
            idx = normalized_categories.index(alias)
            return idx, 1.0, str(categories[idx])

    best_idx: int | None = None
    best_score = 0.0
    for alias in aliases:
        atok = tokens(alias)
        for idx, cat in enumerate(normalized_categories):
            ctok = tokens(cat)
            union = atok | ctok
            jaccard = len(atok & ctok) / len(union) if union else 0.0
            sequence = SequenceMatcher(None, alias, cat).ratio()
            containment = 1.0 if alias and (alias in cat or cat in alias) else 0.0
            score = max(jaccard, 0.7 * sequence, 0.85 * containment)
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_idx is None or best_score < 0.42:
        return None, float(best_score), None
    return best_idx, float(best_score), str(categories[best_idx])


def download_files() -> dict[str, Any]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for filename, spec in FILES.items():
        response = requests.get(spec["url"], timeout=120)
        response.raise_for_status()
        data = response.content
        actual = git_blob_sha1(data)
        if actual != spec["git_blob_sha1"]:
            raise RuntimeError(f"709 file SHA mismatch for {filename}: {actual}")
        (SOURCE / filename).write_bytes(data)
        manifest[filename] = {
            "url": spec["url"], "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha1": actual,
        }
    return manifest


def shared_external_matrix() -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    features = np.load(SOURCE / "features_cat_onehot.npy")
    labels = np.load(SOURCE / "labels.npy").astype(int)
    with open(SOURCE / "enc.pkl", "rb") as handle:
        encoder = pickle.load(handle)
    categories = {name: np.asarray(cat) for name, cat in zip(ENCODER_COLUMNS, encoder.categories_)}
    lengths = [len(categories[name]) for name in ENCODER_COLUMNS]
    offsets = np.cumsum([0] + lengths)
    selected_indices: list[int] = []
    for name in SHARED_COLUMNS:
        pos = ENCODER_COLUMNS.index(name)
        selected_indices.extend(range(int(offsets[pos]), int(offsets[pos + 1])))
    if features.shape[1] != int(offsets[-1]):
        raise RuntimeError(f"Encoder/features mismatch: {features.shape[1]} vs {offsets[-1]}")
    return features[:, selected_indices], labels, {
        "full_shape": list(features.shape), "shared_shape": [len(features), len(selected_indices)],
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "encoder_category_lengths": dict(zip(ENCODER_COLUMNS, lengths)),
        "selected_columns": SHARED_COLUMNS,
    }, categories


def encode_ofet(ofet: pd.DataFrame, categories: dict[str, np.ndarray]) -> tuple[np.ndarray, pd.DataFrame]:
    lengths = [len(categories[name]) for name in SHARED_COLUMNS]
    offsets = np.cumsum([0] + lengths)
    matrix = np.zeros((len(ofet), int(offsets[-1])), dtype=float)
    audit_rows: list[dict[str, Any]] = []
    for field_idx, field in enumerate(SHARED_COLUMNS):
        source_col = OFET_MAPPING[field]
        cats = categories[field]
        start = int(offsets[field_idx])
        for row_pos, value in enumerate(ofet[source_col]):
            idx, score, matched = best_category(field, value, cats)
            if idx is not None:
                matrix[row_pos, start + idx] = 1.0
            audit_rows.append({
                "field": field, "source_column": source_col, "raw_value": value,
                "matched_category": matched, "score": score, "mapped": idx is not None,
            })
    audit = pd.DataFrame(audit_rows)
    return matrix, audit


def choose_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[str, dict[str, Any]]:
    candidates = {
        "logistic_C0.1": LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=SEED),
        "logistic_C1": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED),
        "logistic_C10": LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=2),
    }
    unique = np.unique(groups)
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    scores: dict[str, list[float]] = {name: [] for name in candidates}
    for train_idx, valid_idx in splitter.split(X, y, groups):
        for name, model in candidates.items():
            fitted = model.fit(X[train_idx], y[train_idx])
            pred = fitted.predict(X[valid_idx])
            scores[name].append(float(balanced_accuracy_score(y[valid_idx], pred)))
    summary = {
        name: {"mean_balanced_accuracy": float(np.mean(vals)), "worst_balanced_accuracy": float(np.min(vals))}
        for name, vals in scores.items()
    }
    selected = max(candidates, key=lambda n: (summary[n]["worst_balanced_accuracy"], summary[n]["mean_balanced_accuracy"], n))
    return selected, summary


def build_model(name: str):
    if name == "logistic_C0.1":
        return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=SEED)
    if name == "logistic_C1":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED)
    if name == "logistic_C10":
        return LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced", random_state=SEED)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=2)


def row_bootstrap(y: np.ndarray, pred: np.ndarray, n: int = 20000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        skill = 2.0 * balanced_accuracy_score(y[idx], pred[idx]) - 1.0
        values.append(float(skill))
    arr = np.asarray(values)
    return {
        "row_bootstrap_ci_low": float(np.quantile(arr, 0.025)),
        "row_bootstrap_ci_high": float(np.quantile(arr, 0.975)),
        "row_bootstrap_p_skill_positive": float(np.mean(arr > 0)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    file_manifest = download_files()
    X_external, y_external, external_meta, categories = shared_external_matrix()
    ofet = pd.read_csv(OFET)
    ofet = ofet[ofet["mobility"] > 0].copy().reset_index(drop=True)
    y_train = (ofet["mobility"].to_numpy(float) >= 1.0).astype(int)
    X_train, mapping_audit = encode_ofet(ofet, categories)
    mapping_audit.to_csv(RESULTS / "ofet_to_709_category_mapping.csv", index=False)
    mapping_summary = mapping_audit.groupby("field").agg(
        rows=("mapped", "size"), mapped=("mapped", "sum"), mean_score=("score", "mean")
    ).reset_index()
    mapping_summary["mapping_fraction"] = mapping_summary["mapped"] / mapping_summary["rows"]
    mapping_summary.to_csv(RESULTS / "ofet_to_709_mapping_summary.csv", index=False)

    selected, inner_scores = choose_model(X_train, y_train, ofet["source"].astype(str).to_numpy())
    model = build_model(selected).fit(X_train, y_train)
    pred = model.predict(X_external)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_external)[:, 1]
    else:
        probability = pred.astype(float)

    balanced = float(balanced_accuracy_score(y_external, pred))
    metrics = {
        "selected_model": selected,
        "n_train": int(len(y_train)),
        "n_external": int(len(y_external)),
        "train_sources": int(ofet["source"].nunique()),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "external_class_counts": external_meta["label_counts"],
        "accuracy": float(accuracy_score(y_external, pred)),
        "balanced_accuracy": balanced,
        "balanced_accuracy_skill_vs_0.5": float(2.0 * balanced - 1.0),
        "f1": float(f1_score(y_external, pred)),
        "roc_auc": float(roc_auc_score(y_external, probability)),
        "minimum_field_mapping_fraction": float(mapping_summary["mapping_fraction"].min()),
        "mean_field_mapping_fraction": float(mapping_summary["mapping_fraction"].mean()),
        "inner_scores": inner_scores,
        **row_bootstrap(y_external, pred),
    }
    metrics["empirical_dataset_signal_pass"] = bool(
        metrics["balanced_accuracy_skill_vs_0.5"] > 0
        and metrics["row_bootstrap_ci_low"] > 0
        and metrics["minimum_field_mapping_fraction"] >= 0.80
    )
    metrics["source_cluster_validation_available"] = False
    metrics["family_labels_available"] = False
    metrics["strict_universal_dataset_gate_eligible"] = False
    metrics["strict_universal_dataset_gate_pass"] = False
    metrics["adjudication"] = (
        "The 709-OFET repository provides an independent dataset-level test of shared device/process categories, "
        "but lacks public DOI/family row mappings. Row-bootstrap evidence cannot substitute for source-cluster and family-hidden validation."
    )
    (RESULTS / "external_709_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true": y_external, "prediction": pred, "probability_high_mobility": probability}).to_csv(
        RESULTS / "external_709_predictions.csv", index=False
    )
    manifest = {"files": file_manifest, "external_meta": external_meta, "encoder_columns": ENCODER_COLUMNS, "shared_columns": SHARED_COLUMNS}
    (RESULTS / "external_709_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
