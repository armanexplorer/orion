from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODELS_RTX6000_DEFAULT = ["ResNet50", "MobileNetV2", "ResNet101", "BERT"]
LAT_US_TO_MS = 1.0 / 1000.0


@dataclass(frozen=True)
class ResultsPaths:
    base: Path
    results: Path
    orion: Path
    mps: Path
    ideal: Path


def resolve_paths(base: Path | str = ".") -> ResultsPaths:
    base_path = Path(base)
    results = base_path / "results"
    return ResultsPaths(
        base=base_path,
        results=results,
        orion=results / "orion",
        mps=results / "mps",
        ideal=results / "ideal",
    )


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def orion_metrics(d: dict) -> dict:
    return {
        "throughput": d.get("throughput"),
        "p50": d.get("p50_latency"),
        "p95": d.get("p95_latency"),
        "p99": d.get("p99_latency"),
    }


def mps_metrics(d: dict, idx: int) -> dict:
    # idx 0 = model0/HP; idx 1 = model1/BE
    return {
        "throughput": d.get(f"throughput-{idx}"),
        "p50": d.get(f"p50-latency-{idx}"),
        "p95": d.get(f"p95-latency-{idx}"),
        "p99": d.get(f"p99-latency-{idx}"),
    }


def load_orion(orion_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    pat = re.compile(r"(.+?)_(.+?)_(\\d+)_(be|hp)\\.json$")
    for p in orion_dir.glob("*.json"):
        m = pat.match(p.name)
        if not m:
            continue
        be, hp, run, kind = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        met = orion_metrics(load_json(p))
        rows.append(
            {
                "be_model": be,
                "hp_model": hp,
                "run": run,
                "client": kind,
                **{f"orion_{k}": v for k, v in met.items()},
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No Orion results parsed under {orion_dir}")
    return df


def load_mps(mps_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    pat = re.compile(r"(.+?)_(.+?)_(\\d+)\\.json$")
    for p in mps_dir.glob("*.json"):
        m = pat.match(p.name)
        if not m:
            continue
        hp, be, run = m.group(1), m.group(2), int(m.group(3))
        d = load_json(p)
        met_hp = mps_metrics(d, 0)
        met_be = mps_metrics(d, 1)
        rows.append(
            {
                "be_model": be,
                "hp_model": hp,
                "run": run,
                **{f"mps_be_{k}": v for k, v in met_be.items()},
                **{f"mps_hp_{k}": v for k, v in met_hp.items()},
                "mps_duration": d.get("duration"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No MPS results parsed under {mps_dir}")
    return df


def load_ideal(ideal_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    pat = re.compile(r"(.+?)_(\\d+)_hp\\.json$")
    for p in ideal_dir.glob("*.json"):
        m = pat.match(p.name)
        if not m:
            continue
        model, run = m.group(1), int(m.group(2))
        met = orion_metrics(load_json(p))
        rows.append({"model": model, "run": run, **{f"ideal_{k}": v for k, v in met.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No Ideal results parsed under {ideal_dir}")
    return df


def build_pair_df(orion_raw: pd.DataFrame, mps_raw: pd.DataFrame, ideal_raw: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (be_model, hp_model, run) with Orion+MPS+Ideal-derived metrics."""

    # Pivot Orion to one row per (be_model, hp_model, run) with separate BE/HP columns
    orion_be = (
        orion_raw[orion_raw["client"] == "be"]
        .drop(columns=["client"])
        .rename(
            columns={
                "orion_throughput": "orion_be_throughput",
                "orion_p50": "orion_be_p50",
                "orion_p95": "orion_be_p95",
                "orion_p99": "orion_be_p99",
            }
        )
    )
    orion_hp = (
        orion_raw[orion_raw["client"] == "hp"]
        .drop(columns=["client"])
        .rename(
            columns={
                "orion_throughput": "orion_hp_throughput",
                "orion_p50": "orion_hp_p50",
                "orion_p95": "orion_hp_p95",
                "orion_p99": "orion_hp_p99",
            }
        )
    )

    df = orion_be.merge(orion_hp, on=["be_model", "hp_model", "run"], how="inner")
    df = df.merge(mps_raw, on=["be_model", "hp_model", "run"], how="inner")

    # Ideal: average across runs per model, and map onto BE/HP roles
    ideal_mean = ideal_raw.groupby("model", as_index=True).mean(numeric_only=True)

    def _map_ideal(series: pd.Series, col: str) -> pd.Series:
        return series.map(ideal_mean[col].to_dict())

    df["ideal_be_throughput"] = _map_ideal(df["be_model"], "ideal_throughput")
    df["ideal_be_p50"] = _map_ideal(df["be_model"], "ideal_p50")
    df["ideal_be_p95"] = _map_ideal(df["be_model"], "ideal_p95")
    df["ideal_be_p99"] = _map_ideal(df["be_model"], "ideal_p99")

    df["ideal_hp_throughput"] = _map_ideal(df["hp_model"], "ideal_throughput")
    df["ideal_hp_p50"] = _map_ideal(df["hp_model"], "ideal_p50")
    df["ideal_hp_p95"] = _map_ideal(df["hp_model"], "ideal_p95")
    df["ideal_hp_p99"] = _map_ideal(df["hp_model"], "ideal_p99")

    missing_ideal = df[df[["ideal_be_throughput", "ideal_hp_throughput"]].isna().any(axis=1)]
    if not missing_ideal.empty:
        pairs = missing_ideal[["be_model", "hp_model"]].drop_duplicates().to_dict("records")
        raise RuntimeError(
            "Missing ideal data for some models; re-run run_ideal.py for these pairs: "
            + ", ".join(f"{p['be_model']}/{p['hp_model']}" for p in pairs)
        )

    # Totals and ratios
    df["orion_total_throughput"] = df["orion_be_throughput"] + df["orion_hp_throughput"]
    df["mps_total_throughput"] = df["mps_be_throughput"] + df["mps_hp_throughput"]
    df["ideal_total_throughput"] = df["ideal_be_throughput"] + df["ideal_hp_throughput"]

    df["orion_total_efficiency"] = df["orion_total_throughput"] / df["ideal_total_throughput"]
    df["mps_total_efficiency"] = df["mps_total_throughput"] / df["ideal_total_throughput"]

    # HP protection: p95 slowdown relative to isolated (Ideal)
    df["orion_hp_p95_slowdown"] = df["orion_hp_p95"] / df["ideal_hp_p95"]
    df["mps_hp_p95_slowdown"] = df["mps_hp_p95"] / df["ideal_hp_p95"]

    # BE cost: throughput fraction of isolated
    df["orion_be_thr_frac_of_ideal"] = df["orion_be_throughput"] / df["ideal_be_throughput"]
    df["mps_be_thr_frac_of_ideal"] = df["mps_be_throughput"] / df["ideal_be_throughput"]

    return df.sort_values(["be_model", "hp_model", "run"]).reset_index(drop=True)


def plot_total_efficiency(df: pd.DataFrame, *, title: str = "Total Throughput Efficiency vs Ideal") -> None:
    plot_df = df.copy()
    plot_df["pair"] = plot_df["be_model"] + " / " + plot_df["hp_model"]
    plot_df = plot_df.sort_values("pair")

    x = range(len(plot_df))
    w = 0.4
    plt.figure(figsize=(14, 5))
    plt.bar([i - w / 2 for i in x], plot_df["orion_total_efficiency"], width=w, label="Orion / Ideal")
    plt.bar([i + w / 2 for i in x], plot_df["mps_total_efficiency"], width=w, label="MPS / Ideal")
    plt.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    plt.xticks(list(x), plot_df["pair"], rotation=45, ha="right")
    plt.ylabel("Total throughput efficiency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hp_p95_slowdown(df: pd.DataFrame, *, title: str = "HP p95 Slowdown vs Ideal") -> None:
    plot_df = df.copy()
    plot_df["pair"] = plot_df["be_model"] + " / " + plot_df["hp_model"]
    plot_df = plot_df.sort_values("pair")

    x = range(len(plot_df))
    w = 0.4
    plt.figure(figsize=(14, 5))
    plt.bar([i - w / 2 for i in x], plot_df["orion_hp_p95_slowdown"], width=w, label="Orion HP p95 / Ideal")
    plt.bar([i + w / 2 for i in x], plot_df["mps_hp_p95_slowdown"], width=w, label="MPS HP p95 / Ideal")
    plt.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.6)
    plt.xticks(list(x), plot_df["pair"], rotation=45, ha="right")
    plt.ylabel("HP p95 slowdown vs Ideal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tradeoff(df: pd.DataFrame, *, title: str = "Tradeoff: BE Progress vs HP Protection") -> None:
    plot_df = df.copy()
    plot_df["pair"] = plot_df["be_model"] + " / " + plot_df["hp_model"]

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["orion_be_thr_frac_of_ideal"], plot_df["orion_hp_p95_slowdown"], label="Orion")
    plt.scatter(plot_df["mps_be_thr_frac_of_ideal"], plot_df["mps_hp_p95_slowdown"], label="MPS")

    for _, r in plot_df.iterrows():
        plt.annotate(
            r["pair"],
            (r["orion_be_thr_frac_of_ideal"], r["orion_hp_p95_slowdown"]),
            fontsize=8,
            alpha=0.55,
        )

    plt.axhline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.axvline(1.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    plt.xlabel("BE throughput / Ideal(BE)")
    plt.ylabel("HP p95 / Ideal(HP)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------- Paper-style CSV/figure helpers --------------------

def mean_std_str(values: Iterable[float]) -> str:
    vals = list(values)
    if not vals:
        return "0/0"
    arr = np.asarray(vals, dtype=float)
    return f"{np.average(arr):.2f}/{np.std(arr):.2f}"


def empty_matrix(models: list[str]) -> pd.DataFrame:
    return pd.DataFrame("0/0", index=models, columns=models)


def load_json_maybe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def build_paper_matrices_from_disk(
    base: Path | str = ".",
    *,
    models: list[str] | None = None,
    runs: list[int] | None = None,
    latency_unit: str = "us",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild the same CSV matrices used by gather_results.py/plot_*.py.

    - Uses p95 latency of HP.
    - For Orion-like methods: reads per-client JSONs.
    - For MPS/Streams: reads single JSONs and uses idx0 as HP, idx1 as BE.

    latency_unit: 'us' (JSON values) or 'ms' (CSV output). The paper scripts label ms;
                 the repo JSONs are typically in microseconds.
    """

    if models is None:
        models = MODELS_RTX6000_DEFAULT
    if runs is None:
        runs = [0]

    base_path = Path(base)
    results_dir = base_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lat_scale = 1.0
    if latency_unit == "ms":
        lat_scale = LAT_US_TO_MS
    elif latency_unit != "us":
        raise ValueError("latency_unit must be 'us' or 'ms'")

    mats: dict[str, dict[str, pd.DataFrame]] = {}

    # ---------------- Ideal (isolated) ----------------
    ideal_lat = empty_matrix(models)
    ideal_be_thr = empty_matrix(models)
    ideal_hp_thr = empty_matrix(models)

    ideal_stats: dict[str, dict[str, float]] = {}
    for model in models:
        thr_vals: list[float] = []
        p95_vals: list[float] = []
        for p in (results_dir / "ideal").glob(f"{model}_*_hp.json"):
            d = load_json_maybe(p)
            if not d:
                continue
            if d.get("throughput") is not None:
                thr_vals.append(float(d["throughput"]))
            if d.get("p95_latency") is not None:
                p95_vals.append(float(d["p95_latency"]) * lat_scale)
        if not thr_vals or not p95_vals:
            raise RuntimeError(f"Missing/empty Ideal results for {model} under {results_dir}/ideal")
        ideal_stats[model] = {
            "throughput_mean": float(np.mean(thr_vals)),
            "throughput_std": float(np.std(thr_vals)),
            "p95_mean": float(np.mean(p95_vals)),
            "p95_std": float(np.std(p95_vals)),
        }

    for be in models:
        for hp in models:
            ideal_lat.at[be, hp] = f"{ideal_stats[hp]['p95_mean']:.2f}/{ideal_stats[hp]['p95_std']:.2f}"
            ideal_hp_thr.at[be, hp] = (
                f"{ideal_stats[hp]['throughput_mean']:.2f}/{ideal_stats[hp]['throughput_std']:.2f}"
            )
            ideal_be_thr.at[be, hp] = (
                f"{ideal_stats[be]['throughput_mean']:.2f}/{ideal_stats[be]['throughput_std']:.2f}"
            )

    mats["ideal"] = {"latency": ideal_lat, "be_throughput": ideal_be_thr, "hp_throughput": ideal_hp_thr}
    ideal_lat.to_csv(results_dir / "ideal_latency.csv")
    ideal_be_thr.to_csv(results_dir / "ideal_be_throughput.csv")
    ideal_hp_thr.to_csv(results_dir / "ideal_hp_throughput.csv")

    # ---------------- MPS / Streams (single JSON per pair) ----------------
    def _build_baseline_pairjson(prefix: str) -> None:
        bdir = results_dir / prefix
        if not bdir.exists():
            return
        lat = empty_matrix(models)
        be_thr = empty_matrix(models)
        hp_thr = empty_matrix(models)

        for be in models:
            for hp in models:
                lat_vals, be_thr_vals, hp_thr_vals = [], [], []
                for run in runs:
                    p = bdir / f"{hp}_{be}_{run}.json"  # HP first, then BE
                    d = load_json_maybe(p)
                    if not d:
                        continue
                    if d.get("p95-latency-0") is not None:
                        lat_vals.append(float(d["p95-latency-0"]) * lat_scale)
                    if d.get("throughput-0") is not None:
                        hp_thr_vals.append(float(d["throughput-0"]))
                    if d.get("throughput-1") is not None:
                        be_thr_vals.append(float(d["throughput-1"]))

                lat.at[be, hp] = mean_std_str(lat_vals)
                hp_thr.at[be, hp] = mean_std_str(hp_thr_vals)
                be_thr.at[be, hp] = mean_std_str(be_thr_vals)

        mats[prefix] = {"latency": lat, "be_throughput": be_thr, "hp_throughput": hp_thr}
        lat.to_csv(results_dir / f"{prefix}_latency.csv")
        be_thr.to_csv(results_dir / f"{prefix}_be_throughput.csv")
        hp_thr.to_csv(results_dir / f"{prefix}_hp_throughput.csv")

    _build_baseline_pairjson("mps")
    _build_baseline_pairjson("streams")

    # ---------------- Orion/Temporal/REEF (per-client JSONs) ----------------
    def _build_orion_like(prefix: str) -> None:
        ddir = results_dir / prefix
        if not ddir.exists():
            return
        lat = empty_matrix(models)
        be_thr = empty_matrix(models)
        hp_thr = empty_matrix(models)

        for be in models:
            for hp in models:
                lat_vals, be_thr_vals, hp_thr_vals = [], [], []
                for run in runs:
                    d_hp = load_json_maybe(ddir / f"{be}_{hp}_{run}_hp.json")
                    d_be = load_json_maybe(ddir / f"{be}_{hp}_{run}_be.json")

                    if d_hp and d_hp.get("p95_latency") is not None:
                        lat_vals.append(float(d_hp["p95_latency"]) * lat_scale)
                    if d_hp and d_hp.get("throughput") is not None:
                        hp_thr_vals.append(float(d_hp["throughput"]))
                    if d_be and d_be.get("throughput") is not None:
                        be_thr_vals.append(float(d_be["throughput"]))

                lat.at[be, hp] = mean_std_str(lat_vals)
                hp_thr.at[be, hp] = mean_std_str(hp_thr_vals)
                be_thr.at[be, hp] = mean_std_str(be_thr_vals)

        mats[prefix] = {"latency": lat, "be_throughput": be_thr, "hp_throughput": hp_thr}
        lat.to_csv(results_dir / f"{prefix}_latency.csv")
        be_thr.to_csv(results_dir / f"{prefix}_be_throughput.csv")
        hp_thr.to_csv(results_dir / f"{prefix}_hp_throughput.csv")

    for prefix in ["temporal", "orion", "reef"]:
        _build_orion_like(prefix)

    return mats


def parse_mean_matrix(csv_path: Path, *, models: list[str]) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.drop(df.columns[0], axis=1)
    df.index = models
    for r in models:
        for c in models:
            df.at[r, c] = float(str(df.at[r, c]).split("/")[0])
    return df.mean()


def parse_std_matrix(csv_path: Path, *, models: list[str]) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.drop(df.columns[0], axis=1)
    df.index = models
    for r in models:
        for c in models:
            df.at[r, c] = float(str(df.at[r, c]).split("/")[0])
    return df.std()


def parse_mean_pair(csv_be: Path, csv_hp: Path, *, models: list[str]) -> tuple[pd.Series, pd.Series]:
    df_be = pd.read_csv(csv_be)
    df_hp = pd.read_csv(csv_hp)
    df_be = df_be.drop(df_be.columns[0], axis=1)
    df_hp = df_hp.drop(df_hp.columns[0], axis=1)
    df_be.index = models
    df_hp.index = models
    for r in models:
        for c in models:
            df_be.at[r, c] = float(str(df_be.at[r, c]).split("/")[0])
            df_hp.at[r, c] = float(str(df_hp.at[r, c]).split("/")[0])
    return df_be.mean(), df_hp.mean()


def parse_std_pair(csv_be: Path, csv_hp: Path, *, models: list[str]) -> tuple[pd.Series, pd.Series]:
    df_be = pd.read_csv(csv_be)
    df_hp = pd.read_csv(csv_hp)
    df_be = df_be.drop(df_be.columns[0], axis=1)
    df_hp = df_hp.drop(df_hp.columns[0], axis=1)
    df_be.index = models
    df_hp.index = models
    for r in models:
        for c in models:
            df_be.at[r, c] = float(str(df_be.at[r, c]).split("/")[0])
            df_hp.at[r, c] = float(str(df_hp.at[r, c]).split("/")[0])
    return df_be.std(), df_hp.std()
