from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor
from scipy.io import loadmat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20260726
ROOT = Path("ofet_hidden_validation")
OFET_PATH = Path("ofet_raw/csv/combined_seed_data__final_formatted.csv")
P3HT_URL = "https://raw.githubusercontent.com/Imperssonator/OFET-Database/master/Functions/OFETDatabase.mat"
P3HT_GIT_BLOB_SHA1 = "2170ef810e53f3f1b546b4c98c7bb5667f1a6f80"

NUMERIC_FEATURES = [
    "mn_kda", "mw_kda", "dispersity", "regioregularity_pct",
    "concentration_mg_ml", "dielectric_thickness_nm", "channel_length_um",
    "channel_width_um", "dielectric_capacitance", "spin_rate_rpm",
    "spin_time_s", "coating_speed", "process_temperature_c",
    "anneal_temperature_c", "anneal_time", "measurement_temperature_c", "vds_v",
]
CATEGORICAL_FEATURES = [
    "carrier_type", "solvent", "gate_material", "dielectric_material",
    "electrode_material", "device_configuration", "substrate_pretreat",
    "deposition_type", "process_environment", "anneal_environment",
    "mobility_regime", "measurement_environment",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def norm_doi(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi\s*:\s*", "", s)
    return s.rstrip(".,; ")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def text(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str).str.strip()


def first_existing(df: pd.DataFrame, names: list[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def normalize_family(value: Any) -> str:
    s = str(value).strip().lower()
    if "dpp" in s and "dtt" in s:
        return "DPP-DTT"
    if "n2200" in s or "ndi2od" in s:
        return "N2200"
    if "p3ht" in s or "3-hexylthiophene" in s:
        return "P3HT"
    return str(value).strip()


def build_ofetdb() -> pd.DataFrame:
    raw = pd.read_csv(OFET_PATH)
    base = pd.DataFrame(index=raw.index)
    base["dataset_id"] = "OFET-db"
    base["record_id"] = "OFETDB_" + raw["sample_id"].astype(str)
    base["source"] = raw["meta.doi"].map(norm_doi)
    base["family"] = raw["common_name"].map(normalize_family)
    base["mn_kda"] = numeric(raw["mn"])
    base["mw_kda"] = numeric(raw["mw"])
    base["dispersity"] = numeric(raw["dispersity"])
    base["regioregularity_pct"] = numeric(raw["regioregularity"])
    base["concentration_mg_ml"] = numeric(raw["concentration"])
    base["solvent"] = text(raw["iupac_name.1"])
    base["gate_material"] = text(raw["gate_material"])
    base["dielectric_material"] = text(raw["dielectric_material"])
    base["dielectric_thickness_nm"] = numeric(raw["dielectric_thickness"])
    base["electrode_material"] = text(raw["electrode_material"])
    base["device_configuration"] = text(raw["electrode_configuration"])
    base["channel_length_um"] = numeric(raw["channel_length"])
    base["channel_width_um"] = numeric(raw["channel_width"])
    base["dielectric_capacitance"] = numeric(raw["dielectric_capacitance"])
    base["substrate_pretreat"] = text(raw["substrate_pretreat.sam.iupac_name"])
    base["deposition_type"] = text(raw["deposition_type"])
    base["spin_rate_rpm"] = numeric(raw["params.spin_rate"])
    base["spin_time_s"] = numeric(raw["params.spin_time"])
    base["coating_speed"] = numeric(raw["params.coating_speed"])
    base["process_environment"] = text(raw["params.environment"])
    base["process_temperature_c"] = numeric(raw["params.temperature"])
    base["anneal_temperature_c"] = numeric(raw["postprocess.annealing.temperature"])
    base["anneal_time"] = numeric(raw["postprocess.annealing.time"])
    base["anneal_environment"] = text(raw["coating_process.annealing.environment"])
    base["measurement_temperature_c"] = numeric(raw["ofet.temperature"])
    base["mobility_regime"] = text(raw["ofet.mobility_regime"])
    base["measurement_environment"] = text(raw["ofet.environment"])
    base["vds_v"] = numeric(raw["ofet.Vds"])

    rows: list[pd.DataFrame] = []
    for carrier, target_col in [("hole", "ofet.hole_mobility"), ("electron", "ofet.electron_mobility")]:
        part = base.copy()
        part["carrier_type"] = carrier
        part["mobility"] = numeric(raw[target_col])
        part = part[part["mobility"] > 0].copy()
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["family"].isin(["P3HT", "DPP-DTT", "N2200"])].copy()
    out["log10_mobility"] = np.log10(out["mobility"])
    out = out.drop_duplicates(subset=["dataset_id", "record_id", "carrier_type", "log10_mobility"])
    return out.reset_index(drop=True)


def download_and_parse_p3ht() -> tuple[pd.DataFrame, dict[str, Any]]:
    response = requests.get(P3HT_URL, timeout=120)
    response.raise_for_status()
    data = response.content
    actual_blob = git_blob_sha1(data)
    if actual_blob != P3HT_GIT_BLOB_SHA1:
        raise RuntimeError(f"P3HT MAT blob mismatch: {actual_blob}")
    source_dir = ROOT / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    mat_path = source_dir / "OFETDatabase.mat"
    mat_path.write_bytes(data)

    records = np.ravel(loadmat(mat_path, simplify_cells=True)["OFET"])
    raw = pd.DataFrame([dict(r) for r in records])
    if len(raw) != 218:
        raise RuntimeError(f"Expected 218 P3HT records, found {len(raw)}")

    out = pd.DataFrame(index=raw.index)
    out["dataset_id"] = "P3HT-218"
    out["record_id"] = [f"P3HT_{i + 1:04d}" for i in range(len(raw))]
    out["source"] = first_existing(raw, ["DOI", "doi"]).map(norm_doi)
    out["family"] = "P3HT"
    out["carrier_type"] = "hole"
    out["mobility"] = numeric(first_existing(raw, ["RTMob", "Mobility", "mobility"]))
    out["mn_kda"] = numeric(first_existing(raw, ["Mn"]))
    out["mw_kda"] = numeric(first_existing(raw, ["Mw"]))
    out["dispersity"] = numeric(first_existing(raw, ["PDI"]))
    out["regioregularity_pct"] = numeric(first_existing(raw, ["RR"]))
    out["concentration_mg_ml"] = numeric(first_existing(raw, ["InitConc"]))
    out["solvent"] = text(first_existing(raw, ["Solv1"]))
    out["gate_material"] = "__MISSING__"
    out["dielectric_material"] = "__MISSING__"
    out["dielectric_thickness_nm"] = np.nan
    out["electrode_material"] = text(first_existing(raw, ["ElectrodeMat"]))
    out["device_configuration"] = text(first_existing(raw, ["OFETConfig"]))
    out["channel_length_um"] = numeric(first_existing(raw, ["ChanLen"]))
    width = numeric(first_existing(raw, ["ChanWid"]))
    out["channel_width_um"] = np.where(width.notna() & (width.abs() < 100), width * 1000.0, width)
    out["dielectric_capacitance"] = np.nan
    out["substrate_pretreat"] = text(first_existing(raw, ["SubsTreat"]))
    out["deposition_type"] = text(first_existing(raw, ["Depo"]))
    out["spin_rate_rpm"] = numeric(first_existing(raw, ["SpinRate"]))
    out["spin_time_s"] = numeric(first_existing(raw, ["SpinTime"]))
    out["coating_speed"] = numeric(first_existing(raw, ["DipRate"]))
    out["process_environment"] = text(first_existing(raw, ["ProcEnv"]))
    out["process_temperature_c"] = np.nan
    out["anneal_temperature_c"] = numeric(first_existing(raw, ["AnnTemp"]))
    out["anneal_time"] = numeric(first_existing(raw, ["AnnTime"]))
    out["anneal_environment"] = text(first_existing(raw, ["AnnCool"]))
    out["measurement_temperature_c"] = np.nan
    out["mobility_regime"] = text(first_existing(raw, ["OFETReg"]))
    out["measurement_environment"] = text(first_existing(raw, ["MobEnv"]))
    out["vds_v"] = numeric(first_existing(raw, ["Vds"]))
    out = out[out["mobility"] > 0].copy()
    out["log10_mobility"] = np.log10(out["mobility"])
    manifest = {
        "url": P3HT_URL,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": actual_blob,
        "records": int(len(out)),
        "source_groups": int(out["source"].nunique()),
        "raw_columns": list(raw.columns),
    }
    return out.reset_index(drop=True), manifest


def prepare_native(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    a = train[FEATURES].copy()
    b = test[FEATURES].copy()
    cat_indices: list[int] = []
    for i, col in enumerate(FEATURES):
        if col in CATEGORICAL_FEATURES:
            a[col] = text(a[col])
            b[col] = text(b[col])
            cat_indices.append(i)
        else:
            a[col] = numeric(a[col])
            b[col] = numeric(b[col])
            median = a[col].median()
            if pd.isna(median):
                median = 0.0
            a[col] = a[col].fillna(median)
            b[col] = b[col].fillna(median)
    return a, b, cat_indices


def ridge_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def predict_candidate(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y = train["log10_mobility"].to_numpy(float)
    if name.startswith("ridge_"):
        alpha = float(name.split("_", 1)[1])
        model = ridge_pipeline(alpha)
        model.fit(train[FEATURES], y)
        return model.predict(test[FEATURES])
    a, b, cat_indices = prepare_native(train, test)
    if name == "cat_shallow":
        params = dict(iterations=250, depth=4, learning_rate=0.035, l2_leaf_reg=8)
    elif name == "cat_medium":
        params = dict(iterations=350, depth=6, learning_rate=0.025, l2_leaf_reg=12)
    else:
        raise ValueError(name)
    model = CatBoostRegressor(
        **params, loss_function="MAE", verbose=False, random_seed=SEED,
        random_strength=0.5, allow_writing_files=False, thread_count=2,
    )
    model.fit(a, y, cat_features=cat_indices)
    return model.predict(b)


CANDIDATES = ["ridge_1", "ridge_10", "ridge_100", "cat_shallow", "cat_medium"]


def inner_select(train: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    groups = train["source"].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 3:
        return "ridge_10", {}
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    scores: dict[str, list[float]] = {name: [] for name in CANDIDATES}
    for tr_idx, va_idx in splitter.split(train, groups=groups):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        baseline = np.full(len(va), tr["log10_mobility"].median())
        baseline_mae = mean_absolute_error(va["log10_mobility"], baseline)
        for name in CANDIDATES:
            pred = predict_candidate(name, tr, va)
            ratio = mean_absolute_error(va["log10_mobility"], pred) / (baseline_mae + 1e-12)
            scores[name].append(float(ratio))
    summary = {
        name: {"mean_ratio": float(np.mean(vals)), "worst_ratio": float(np.max(vals))}
        for name, vals in scores.items()
    }
    selected = min(CANDIDATES, key=lambda n: (summary[n]["worst_ratio"], summary[n]["mean_ratio"], n))
    return selected, summary


def cluster_bootstrap(test: pd.DataFrame, pred: np.ndarray, baseline: np.ndarray, n: int = 20000) -> dict[str, float]:
    groups = test["source"].astype(str).to_numpy()
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    y = test["log10_mobility"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    for _ in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices[g] for g in sampled])
        base_mae = np.mean(np.abs(y[idx] - baseline[idx]))
        model_mae = np.mean(np.abs(y[idx] - pred[idx]))
        if base_mae > 0:
            values.append(1.0 - model_mae / base_mae)
    arr = np.asarray(values)
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_skill_positive": float(np.mean(arr > 0)),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    overlap = set(train["source"]) & set(test["source"])
    if overlap:
        raise RuntimeError(f"Source overlap in {label}: {sorted(overlap)[:5]}")
    selected, inner_scores = inner_select(train)
    pred = predict_candidate(selected, train, test)
    baseline = np.full(len(test), train["log10_mobility"].median())
    y = test["log10_mobility"].to_numpy(float)
    model_mae = mean_absolute_error(y, pred)
    baseline_mae = mean_absolute_error(y, baseline)
    skill = 1.0 - model_mae / baseline_mae
    boot = cluster_bootstrap(test, pred, baseline)
    result = {
        "test": label,
        "selected_model": selected,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_sources": int(train["source"].nunique()),
        "test_sources": int(test["source"].nunique()),
        "train_families": sorted(train["family"].unique().tolist()),
        "test_families": sorted(test["family"].unique().tolist()),
        "source_overlap": 0,
        "model_mae": float(model_mae),
        "baseline_mae": float(baseline_mae),
        "skill": float(skill),
        **boot,
        "point_pass": bool(skill > 0),
        "ci_pass": bool(boot["ci_low"] > 0),
        "strict_pass": bool(skill > 0 and boot["ci_low"] > 0),
        "inner_scores": inner_scores,
    }
    predictions = test[["dataset_id", "record_id", "source", "family", "carrier_type", "log10_mobility"]].copy()
    predictions["test"] = label
    predictions["selected_model"] = selected
    predictions["prediction"] = pred
    predictions["baseline_prediction"] = baseline
    return result, predictions


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data"
    result_dir = ROOT / "results"
    data_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    ofet = build_ofetdb()
    p3ht, p3ht_manifest = download_and_parse_p3ht()
    ofet.to_csv(data_dir / "ofetdb_mobility_harmonized.csv", index=False)
    p3ht.to_csv(data_dir / "p3ht_218_harmonized.csv", index=False)

    profile = pd.concat([
        ofet.groupby(["dataset_id", "family", "carrier_type"]).agg(
            rows=("record_id", "size"), sources=("source", "nunique"),
            mobility_min=("mobility", "min"), mobility_median=("mobility", "median"), mobility_max=("mobility", "max"),
        ).reset_index(),
        p3ht.groupby(["dataset_id", "family", "carrier_type"]).agg(
            rows=("record_id", "size"), sources=("source", "nunique"),
            mobility_min=("mobility", "min"), mobility_median=("mobility", "median"), mobility_max=("mobility", "max"),
        ).reset_index(),
    ], ignore_index=True)
    profile.to_csv(result_dir / "dataset_profile.csv", index=False)

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    # Rotating family-hidden tests within OFET-db. Entire DOI groups of the held family are absent from training.
    for held_family in ["P3HT", "DPP-DTT", "N2200"]:
        test = ofet[ofet["family"] == held_family].copy()
        train = ofet[ofet["family"] != held_family].copy()
        train = train[~train["source"].isin(set(test["source"]))].copy()
        result, pred = evaluate(train, test, f"OFET-db leave-{held_family}-out")
        result["validation_type"] = "family_hidden"
        results.append(result)
        predictions.append(pred)

    # Locked dataset + family hidden test: no P3HT records or overlapping DOI in development data.
    external_test = p3ht.copy()
    external_train = ofet[ofet["family"].isin(["DPP-DTT", "N2200"])].copy()
    external_train = external_train[~external_train["source"].isin(set(external_test["source"]))].copy()
    result, pred = evaluate(external_train, external_test, "OFET-db non-P3HT -> external P3HT-218")
    result["validation_type"] = "dataset_and_family_hidden"
    results.append(result)
    predictions.append(pred)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(result_dir / "hidden_validation_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(result_dir / "hidden_validation_predictions.csv", index=False)

    family_results = result_frame[result_frame["validation_type"] == "family_hidden"]
    external_result = result_frame[result_frame["validation_type"] == "dataset_and_family_hidden"]
    gate = {
        "family_rotation_all_pass": bool(family_results["strict_pass"].all()),
        "dataset_and_family_hidden_pass": bool(external_result["strict_pass"].all()),
        "universal_mobility_gate_pass": bool(family_results["strict_pass"].all() and external_result["strict_pass"].all()),
        "acceptance_rule": "skill > 0 and source-cluster bootstrap 95% CI lower bound > 0 for every held family and the external dataset+family test",
        "note": "Retrospective locked external validation; the P3HT dataset was known to the wider project before this split and is not claimed as pristine prospective data.",
    }
    (result_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    manifest = {
        "ofetdb_rows": int(len(ofet)),
        "ofetdb_sources": int(ofet["source"].nunique()),
        "ofetdb_families": sorted(ofet["family"].unique().tolist()),
        "p3ht_external": p3ht_manifest,
        "features": FEATURES,
        "excluded_from_predictors": ["dataset_id", "record_id", "source", "family", "mobility", "log10_mobility", "year", "DOI", "Ion/Ioff", "Vth", "subthreshold_swing"],
        "seed": SEED,
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result_frame[["test", "selected_model", "n_train", "n_test", "skill", "ci_low", "ci_high", "strict_pass"]].to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
