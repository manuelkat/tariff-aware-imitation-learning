from __future__ import annotations

"""TAIL (Tariff-Aware Imitation Learning) runner.

This repository was consolidated to a small number of Python files (excluding tests).
`tail.py` is the canonical entrypoint for the paper/data-generation IL pipeline.

Run:
  python tail.py --config configs/<your_config>.yaml

All outputs are written append-only under results/runs/<RUN_ID>/.
"""

import argparse
import concurrent.futures as cf
import copy
import datetime as _dt
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Allow importing the consolidated implementation under ./src
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required (see requirements.txt)") from e

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import energy_il as eil


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def utc_run_id() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(int(seed))


def _parse_worker_setting(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"auto", "max"}:
            return None
        try:
            parsed = int(float(lowered))
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    if isinstance(value, (int, float)):
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _resolve_parallelism(*, il_cfg: dict[str, Any], n_buildings: int) -> tuple[int, int, int]:
    cpu_cores = max(1, int(os.cpu_count() or 1))
    raw_max_workers = _parse_worker_setting(il_cfg.get("max_building_workers"))
    raw_per_run_workers = _parse_worker_setting(il_cfg.get("parallel_workers", il_cfg.get("per_run_workers")))

    if raw_max_workers is None and raw_per_run_workers is None:
        max_workers = min(int(n_buildings), int(cpu_cores))
        per_run_workers = max(1, int(cpu_cores // max_workers))
    elif raw_max_workers is None:
        per_run_workers = max(1, int(raw_per_run_workers))
        max_workers = min(int(n_buildings), max(1, int(cpu_cores // per_run_workers)))
    elif raw_per_run_workers is None:
        max_workers = min(int(n_buildings), int(raw_max_workers))
        per_run_workers = max(1, int(cpu_cores // max_workers))
    else:
        max_workers = min(int(n_buildings), int(raw_max_workers))
        per_run_workers = max(1, int(raw_per_run_workers))

    max_workers = max(1, int(max_workers))
    per_run_workers = max(1, min(int(per_run_workers), int(cpu_cores)))
    return max_workers, per_run_workers, cpu_cores


def _configure_run_logging(*, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if getattr(root, "_tail_configured", False):
        return

    fmt = logging.Formatter(fmt="%(asctime)sZ | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)
    setattr(root, "_tail_configured", True)


class _LogPrefixFilter(logging.Filter):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        record.msg = f"{self._prefix}{msg}"
        record.args = ()
        return True


def _configure_worker_logging(*, run_dir: Path, log_queue: mp.Queue | None, building_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(fmt="%(asctime)sZ | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if log_queue is not None:
        qh = logging.handlers.QueueHandler(log_queue)
        qh.setLevel(logging.INFO)
        qh.addFilter(_LogPrefixFilter(prefix=f"[building {building_id}] "))
        root.addHandler(qh)


@dataclass
class PipelineTracker:
    run_dir: Path
    cfg: Mapping[str, Any]
    steps: dict[str, Any]
    status: str = "running"
    started_at: str = ""

    def __post_init__(self) -> None:
        self.started_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        self.steps = {}
        self._write()

    def _write(self) -> None:
        write_json(
            self.run_dir / "status.json",
            {
                "status": self.status,
                "started_at": self.started_at,
                "steps": self.steps,
            },
        )

    def start_step(self, name: str, **meta: Any) -> None:
        self.steps[name] = {"status": "running", "started_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(), **meta}
        self._write()

    def end_step(self, name: str, *, ok: bool = True, **meta: Any) -> None:
        step = dict(self.steps.get(name, {}))
        step.update({"status": "completed" if ok else "failed", "ended_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(), **meta})
        self.steps[name] = step
        self._write()

    def update_step(self, name: str, **meta: Any) -> None:
        step = dict(self.steps.get(name, {}))
        step.update(meta)
        self.steps[name] = step
        self._write()

    def mark_completed(self) -> None:
        self.status = "completed"
        self._write()

    def mark_failed(self, msg: str) -> None:
        self.status = "failed"
        self.steps["error"] = {"message": str(msg)}
        self._write()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAIL IL pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit run directory (absolute, or relative to results/runs).",
    )
    p.add_argument(
        "--make-figures-only",
        action="store_true",
        help="Generate figures for an existing run dir using its metrics artifacts, without re-running the pipeline.",
    )
    p.add_argument(
        "--force-figures",
        action="store_true",
        help="Regenerate figures even if they already exist (overwrite).",
    )
    p.add_argument(
        "--detach",
        action="store_true",
        help="Re-launch this process in the background (nohup+setsid) so it survives SSH disconnects. "
             "stdout/stderr go to <run_dir>/nohup.out.  The parent prints the PID and exits immediately.",
    )
    return p.parse_args()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _compute_running_peak_kw(*, grid_kw: np.ndarray, is_peak_window: np.ndarray) -> np.ndarray:
    """Compute running monthly-peak (single segment) consistent with PEAK_WINDOW_MONTHLY_MAX.

    We mirror the env/state update rule: only update the max during the peak window.
    Returns an array of length T+1 (state-like), with peak[0]=0.
    """
    g = np.asarray(grid_kw, dtype=float)
    w = np.asarray(is_peak_window, dtype=int).astype(bool)
    T = int(g.shape[0])
    peak = np.zeros(T + 1, dtype=float)
    for t in range(T):
        peak[t + 1] = max(float(peak[t]), float(g[t])) if bool(w[t]) else float(peak[t])
    return peak


def _assert_bill_consistency(bill: eil.BillBreakdown, *, tol: float = 1e-6) -> None:
    lhs = float(bill.bill_total)
    rhs = float(bill.energy_cost) + float(bill.demand_cost)
    if not np.isfinite(lhs) or not np.isfinite(rhs) or abs(lhs - rhs) > tol:
        raise ValueError("Bill breakdown inconsistent")


def _assert_peak_consistency(
    *,
    bill: eil.BillBreakdown,
    grid_kw: np.ndarray,
    is_peak_window: np.ndarray,
    tol: float = 1e-6,
) -> None:
    g = np.asarray(grid_kw, dtype=float)
    peak = np.asarray(is_peak_window, dtype=int).astype(bool)
    if not np.any(peak):
        expected = 0.0
    else:
        expected = float(np.max(g[peak]))
    if abs(float(bill.peak_kw) - expected) > tol:
        raise ValueError("Peak kW inconsistent with peak-window max")


def _month_slices(timestamps_utc: pd.DatetimeIndex, *, timezone: str) -> list[tuple[int, slice]]:
    tz = ZoneInfo(str(timezone))
    if timestamps_utc.tz is None:
        raise ValueError("timestamps_utc must be timezone-aware")
    ts_local = timestamps_utc.tz_convert(tz)
    months = ts_local.month.to_numpy(dtype=int)
    if months.size == 0:
        return []
    boundaries = np.where(months[1:] != months[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=int), boundaries])
    ends = np.concatenate([boundaries, np.array([months.size], dtype=int)])
    out: list[tuple[int, slice]] = []
    for s, e in zip(starts, ends):
        m = int(months[int(s)])
        out.append((m, slice(int(s), int(e))))
    return out


def _months_from_idx(
    timestamps_utc: pd.DatetimeIndex,
    *,
    timezone: str,
    idx: np.ndarray,
) -> set[int]:
    if idx.size == 0:
        return set()
    tz = ZoneInfo(str(timezone))
    ts_local = timestamps_utc.tz_convert(tz)
    months = ts_local.month.to_numpy(dtype=int)
    return set(int(m) for m in np.unique(months[idx]))


def _max_energy_window_kwh(
    load_kw: np.ndarray,
    *,
    dt_hours: float,
    window_hours: float,
) -> float:
    load_kw = np.asarray(load_kw, dtype=float)
    if load_kw.size == 0:
        return 0.0
    w_steps = max(1, int(round(float(window_hours) / float(dt_hours))))
    energy_kwh = load_kw * float(dt_hours)
    if energy_kwh.size < w_steps:
        return float(np.sum(energy_kwh))
    kernel = np.ones(w_steps, dtype=float)
    window_sums = np.convolve(energy_kwh, kernel, mode="valid")
    return float(np.max(window_sums))


def _make_figures_from_metrics_artifacts(*, run_dir: Path, force: bool = False) -> None:
    log = logging.getLogger("tail")

    metrics_dir = run_dir / "metrics"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = metrics_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics.csv under {metrics_dir}")

    def _should_write(p: Path) -> bool:
        return force or not p.exists()

    # Prefer headless backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- fig_bars.pdf ----
    try:
        dfm = pd.read_csv(metrics_csv)
        if not dfm.empty:
            tariffs = sorted(dfm["tariff"].dropna().unique().tolist())
            model_order = [
                "no_battery",
                "bc_baseline",
                "bc_tariff_aware",
                "rule_tariff_aware",
                "dagger_baseline",
                "dagger_tariff_aware",
                "teacher",
            ]
            # Append any teacher_<noisy_variant> models
            noisy_teacher_models = sorted([
                m for m in dfm["model"].dropna().unique()
                if m.startswith("teacher_") and m != "teacher"
            ])
            model_order.extend(noisy_teacher_models)
            models_present = [m for m in model_order if (dfm["model"] == m).any()]

            # Try loading aggregate data for per-building error bars
            _agg_csv = run_dir / "aggregate" / "aggregate_metrics_by_tariff_model.csv"
            _df_agg: pd.DataFrame | None = None
            if _agg_csv.exists():
                try:
                    _df_agg = pd.read_csv(_agg_csv)
                    if _df_agg.empty:
                        _df_agg = None
                except Exception:
                    _df_agg = None

            def _pivot(col: str) -> np.ndarray:
                out = np.full((len(models_present), len(tariffs)), np.nan, dtype=float)
                for i, m in enumerate(models_present):
                    for j, t in enumerate(tariffs):
                        sel = dfm[(dfm["model"] == m) & (dfm["tariff"] == t)]
                        if not sel.empty:
                            out[i, j] = _safe_float(sel.iloc[0][col], default=np.nan)
                return out

            def _pivot_agg_std(col_std: str) -> np.ndarray | None:
                if _df_agg is None or col_std not in _df_agg.columns:
                    return None
                out = np.full((len(models_present), len(tariffs)), 0.0, dtype=float)
                for i, m in enumerate(models_present):
                    for j, t in enumerate(tariffs):
                        sel = _df_agg[(_df_agg["model"] == m) & (_df_agg["tariff"] == t)]
                        if not sel.empty:
                            out[i, j] = _safe_float(sel.iloc[0][col_std], default=0.0)
                return out

            savings = _pivot("savings_vs_no_battery") * 100.0  # percent
            peaks = _pivot("peak_kw")
            savings_std = _pivot_agg_std("savings_vs_no_battery_std")
            if savings_std is not None:
                savings_std = savings_std * 100.0
            peaks_std = _pivot_agg_std("peak_kw_std")

            x = np.arange(len(models_present))
            width = min(0.8 / max(len(tariffs), 1), 0.35)

            fig, axes = plt.subplots(2, 1, figsize=(max(11, 2 * len(models_present)), 7), sharex=True)
            ax0, ax1 = axes
            ax0.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
            for j, t in enumerate(tariffs):
                offset = (j - (len(tariffs) - 1) / 2.0) * width
                _yerr0 = savings_std[:, j] if savings_std is not None else None
                ax0.bar(x + offset, savings[:, j], width=width, label=t,
                        yerr=_yerr0, capsize=2, error_kw={"linewidth": 0.8})
            ax0.set_title("Savings vs no_battery")
            ax0.set_ylabel("%")
            ax0.grid(True, axis="y", alpha=0.2)
            ax0.legend(loc="best", fontsize="x-small", ncol=max(1, len(tariffs) // 3))

            for j, t in enumerate(tariffs):
                offset = (j - (len(tariffs) - 1) / 2.0) * width
                _yerr1 = peaks_std[:, j] if peaks_std is not None else None
                ax1.bar(x + offset, peaks[:, j], width=width, label=t,
                        yerr=_yerr1, capsize=2, error_kw={"linewidth": 0.8})
            ax1.set_title("Peak demand (within peak window)")
            ax1.set_ylabel("kW")
            ax1.grid(True, axis="y", alpha=0.2)
            ax1.set_xticks(x)
            ax1.set_xticklabels(models_present, rotation=25, ha="right", fontsize="small")

            fig.tight_layout()
            out_path = fig_dir / "fig_bars.pdf"
            if _should_write(out_path):
                fig.savefig(out_path)
                log.info("Wrote %s", str(out_path))
            plt.close(fig)
    except Exception as e:
        log.info("fig_bars generation skipped: %s", str(e))

    # ---- fig_timeseries.pdf ----
    try:
        npz_files = sorted(metrics_dir.glob("eval_rollouts_*.npz"))
        if npz_files:
            fig, axes = plt.subplots(len(npz_files), 2, figsize=(12, 3.5 * len(npz_files)), sharex=False)
            if len(npz_files) == 1:
                axes = np.asarray([axes])

            for i, npz_path in enumerate(npz_files):
                label = npz_path.stem.replace("eval_rollouts_", "")
                z = np.load(npz_path, allow_pickle=True)
                load_kw_fig = np.asarray(z["load_kw"], dtype=float)
                is_peak = np.asarray(z["is_peak_window"], dtype=int).astype(bool)
                dt_hours = float(np.asarray(z["dt_hours"], dtype=float).reshape(-1)[0])
                th = np.arange(load_kw_fig.shape[0], dtype=float) * dt_hours

                keys_pref = [
                    ("no_battery", "grid_no_battery"),
                    ("bc_baseline", "grid_bc_baseline"),
                    ("bc_tariff_aware", "grid_bc_tariff_aware"),
                    ("rule_tariff_aware", "grid_rule_tariff_aware"),
                    ("dagger_baseline", "grid_dagger_baseline"),
                    ("dagger_tariff_aware", "grid_dagger_tariff_aware"),
                    ("teacher", "grid_teacher"),
                ]
                series = [(m, k) for (m, k) in keys_pref if k in z.files]
                keep = {"no_battery", "bc_tariff_aware", "rule_tariff_aware", "dagger_tariff_aware", "teacher"}
                series = [(m, k) for (m, k) in series if m in keep] or series

                axg = axes[i, 0]
                axb = axes[i, 1]
                axg.plot(th, load_kw_fig, color="0.6", linewidth=1.0, label="load")
                for m, k in series:
                    grid = np.asarray(z[k], dtype=float)
                    axg.plot(th, grid, linewidth=1.5, label=m)

                ymax = float(np.nanmax(load_kw_fig)) if load_kw_fig.size else 1.0
                axg.fill_between(th, 0.0, ymax, where=is_peak, alpha=0.08, color="tab:red")
                axg.set_title(f"Grid power ({label})")
                axg.set_xlabel("hours")
                axg.set_ylabel("kW")
                axg.grid(True, alpha=0.2)
                axg.legend(loc="best", ncol=2, fontsize="x-small")

                for m, k in series:
                    grid = np.asarray(z[k], dtype=float)
                    p_batt = load_kw_fig - grid
                    axb.plot(th, p_batt, linewidth=1.5, label=m)
                axb.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
                axb.fill_between(th, 0.0, float(np.nanmax(np.abs(load_kw_fig))) if load_kw_fig.size else 1.0, where=is_peak, alpha=0.08, color="tab:red")
                axb.set_title(f"Implied battery power = load-grid ({label})")
                axb.set_xlabel("hours")
                axb.set_ylabel("kW")
                axb.grid(True, alpha=0.2)
                axb.legend(loc="best", ncol=2, fontsize="x-small")

            fig.tight_layout()
            out_path = fig_dir / "fig_timeseries.pdf"
            if _should_write(out_path):
                fig.savefig(out_path)
                log.info("Wrote %s", str(out_path))
            plt.close(fig)
    except Exception as e:
        log.info("fig_timeseries generation skipped: %s", str(e))

    # ---- extra: fig_economics.pdf (cumulative energy + demand + total) ----
    try:
        ts_csvs = sorted(metrics_dir.glob("eval_timeseries_*.csv"))
        if ts_csvs:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_path = fig_dir / "fig_economics.pdf"
            if _should_write(out_path):
                fig, axes = plt.subplots(len(ts_csvs), 1, figsize=(12, 3.8 * len(ts_csvs)), sharex=False)
                if len(ts_csvs) == 1:
                    axes = [axes]

                for ax, csv_path in zip(axes, ts_csvs, strict=False):
                    label = csv_path.stem.replace("eval_timeseries_", "")
                    df = pd.read_csv(csv_path)
                    models = [
                        "no_battery",
                        "bc_baseline",
                        "bc_tariff_aware",
                        "rule_tariff_aware",
                        "dagger_tariff_aware",
                        "teacher",
                    ]
                    for m in models:
                        col = f"bill_total_cum_{m}"
                        if col in df.columns:
                            ax.plot(df["t"].to_numpy(), df[col].to_numpy(), label=m)
                    ax.set_title(f"Cumulative total bill proxy over time ({label})")
                    ax.set_xlabel("t")
                    ax.set_ylabel("$")
                    ax.grid(True, alpha=0.2)
                    ax.legend(loc="best", ncol=2, fontsize="x-small")

                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                log.info("Wrote %s", str(out_path))
    except Exception as e:
        log.info("fig_economics generation skipped: %s", str(e))

    # ---- extra: fig_battery.pdf (SOC in % + P_batt for TA vs teacher) ----
    try:
        ts_csvs = sorted(metrics_dir.glob("eval_timeseries_*.csv"))
        if ts_csvs:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Read E_max for SoC % conversion
            e_max_kwh = None
            batt_json = run_dir / "battery" / "battery_sizing.json"
            if batt_json.exists():
                try:
                    with open(batt_json) as _bf:
                        e_max_kwh = float(json.loads(_bf.read())["E_max_kwh"])
                except Exception:
                    pass

            out_path = fig_dir / "fig_battery.pdf"
            if _should_write(out_path):
                fig, axes = plt.subplots(len(ts_csvs), 2, figsize=(12, 3.8 * len(ts_csvs)), sharex=False)
                if len(ts_csvs) == 1:
                    axes = np.asarray([axes])

                for i, csv_path in enumerate(ts_csvs):
                    label = csv_path.stem.replace("eval_timeseries_", "")
                    df = pd.read_csv(csv_path)
                    ax_soc = axes[i, 0]
                    ax_p = axes[i, 1]
                    for m in ["bc_tariff_aware", "rule_tariff_aware", "dagger_tariff_aware", "teacher"]:
                        soc = f"soc_kwh_{m}"
                        p = f"P_batt_kw_safe_{m}" if m != "teacher" else "P_batt_kw_teacher"
                        soc_pct_col = f"soc_pct_{m}"
                        if soc_pct_col in df.columns:
                            ax_soc.plot(df["t"].to_numpy(), df[soc_pct_col].to_numpy(dtype=float), label=m)
                        elif soc in df.columns:
                            soc_vals = df[soc].to_numpy(dtype=float)
                            if e_max_kwh and e_max_kwh > 0:
                                soc_vals = soc_vals / e_max_kwh * 100.0
                            ax_soc.plot(df["t"].to_numpy(), soc_vals, label=m)
                        if p in df.columns:
                            ax_p.plot(df["t"].to_numpy(), df[p].to_numpy(), label=m)
                    ax_soc.set_title(f"SOC ({label})")
                    ax_soc.set_xlabel("t")
                    ax_soc.set_ylabel("SoC (%)")
                    ax_soc.grid(True, alpha=0.2)
                    ax_soc.legend(loc="best", fontsize="x-small")

                    ax_p.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
                    ax_p.set_title(f"Battery power ({label})")
                    ax_p.set_xlabel("t")
                    ax_p.set_ylabel("kW")
                    ax_p.grid(True, alpha=0.2)
                    ax_p.legend(loc="best", fontsize="x-small")

                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                log.info("Wrote %s", str(out_path))
    except Exception as e:
        log.info("fig_battery generation skipped: %s", str(e))

    # ---- fig_per_building_savings.pdf ----
    try:
        monthly_csv = metrics_dir / "metrics_monthly.csv"
        if monthly_csv.exists():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_path = fig_dir / "fig_per_building_savings.pdf"
            if _should_write(out_path):
                df_m = pd.read_csv(monthly_csv)
                if "savings_vs_no_battery" not in df_m.columns and "bill_total" in df_m.columns:
                    nb = df_m[df_m["model"] == "no_battery"][
                        ["family", "tariff", "month", "bill_total"]
                    ].rename(columns={"bill_total": "bill_nb"})
                    df_m = df_m.merge(nb, on=["family", "tariff", "month"], how="left")
                    df_m["savings_vs_no_battery"] = (df_m["bill_nb"] - df_m["bill_total"]) / df_m["bill_nb"]

                if "savings_vs_no_battery" in df_m.columns:
                    models_show = ["bc_baseline", "bc_tariff_aware", "rule_tariff_aware",
                                   "dagger_baseline", "dagger_tariff_aware", "teacher"]
                    df_sub = df_m[df_m["model"].isin(models_show)].copy()
                    if not df_sub.empty:
                        families = sorted(df_sub["family"].dropna().unique()) if "family" in df_sub.columns else ["all"]
                        n_fam = max(len(families), 1)
                        fig, axes = plt.subplots(1, n_fam, figsize=(5 * n_fam, 5), sharey=True, squeeze=False)
                        for j, fam in enumerate(families):
                            ax = axes[0, j]
                            dff = df_sub[df_sub["family"] == fam] if "family" in df_sub.columns else df_sub
                            dff.boxplot(column="savings_vs_no_battery", by="model", ax=ax, rot=30, fontsize="small")
                            ax.set_title(f"Savings ({fam})")
                            ax.set_xlabel("")
                            ax.set_ylabel("savings vs no_battery")
                            ax.grid(True, axis="y", alpha=0.2)
                        fig.suptitle("Per-building monthly savings distribution", fontsize=12)
                        fig.tight_layout()
                        fig.savefig(out_path)
                        plt.close(fig)
                        log.info("Wrote %s", str(out_path))
    except Exception as e:
        log.info("fig_per_building_savings generation skipped: %s", str(e))

    # ---- fig_forecast_robustness.pdf ----
    try:
        if metrics_csv.exists():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_path = fig_dir / "fig_forecast_robustness.pdf"
            if _should_write(out_path):
                dfr = pd.read_csv(metrics_csv)
                teacher_models = sorted([
                    m for m in dfr["model"].dropna().unique()
                    if m.startswith("teacher") and m != "no_battery"
                ])
                if len(teacher_models) >= 2 and "bill_total" in dfr.columns:
                    families = sorted(dfr["family"].dropna().unique()) if "family" in dfr.columns else ["all"]
                    tariffs = sorted(dfr["tariff"].dropna().unique()) if "tariff" in dfr.columns else ["all"]
                    fig, axes = plt.subplots(1, max(len(tariffs), 1), figsize=(5 * max(len(tariffs), 1), 4.5), sharey=True, squeeze=False)
                    for ti, tname in enumerate(tariffs):
                        ax = axes[0, ti]
                        dft = dfr[dfr["tariff"] == tname] if "tariff" in dfr.columns else dfr
                        xs = []
                        ys = []
                        labels = []
                        for m in teacher_models:
                            row = dft[dft["model"] == m]
                            if not row.empty:
                                lbl = m.replace("teacher_", "").replace("teacher", "0.0")
                                xs.append(lbl)
                                ys.append(float(row["bill_total"].iloc[0]))
                                labels.append(m)
                        if xs:
                            ax.plot(xs, ys, "o-", linewidth=1.5, markersize=5)
                            ax.set_title(f"Tariff {tname}")
                            ax.set_xlabel("noise variant")
                            ax.set_ylabel("bill ($)")
                            ax.grid(True, alpha=0.2)
                            ax.tick_params(axis="x", rotation=25)
                    fig.suptitle("Forecast robustness: teacher cost vs noise level", fontsize=12)
                    fig.tight_layout()
                    fig.savefig(out_path)
                    plt.close(fig)
                    log.info("Wrote %s", str(out_path))
    except Exception as e:
        log.info("fig_forecast_robustness generation skipped: %s", str(e))

    # ---- fig_early_stopping_consistency.pdf ----
    try:
        cost_csvs = sorted(metrics_dir.glob("train_costs*.csv"))
        if cost_csvs:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_path = fig_dir / "fig_early_stopping_consistency.pdf"
            if _should_write(out_path):
                fig, axes = plt.subplots(len(cost_csvs), 1, figsize=(8, 3.5 * max(len(cost_csvs), 1)), squeeze=False)
                for i, csv_path in enumerate(cost_csvs):
                    ax = axes[i, 0]
                    label = csv_path.stem.replace("train_costs", "").strip("_") or "default"
                    dfc = pd.read_csv(csv_path)
                    cost_col = "bill_cost" if "bill_cost" in dfc.columns else ("cost_total" if "cost_total" in dfc.columns else None)
                    x_col = "epoch" if "epoch" in dfc.columns else ("eval_idx" if "eval_idx" in dfc.columns else None)
                    if x_col is not None and cost_col is not None:
                        for model_name, dfg in dfc.groupby("model", sort=False):
                            ax.plot(dfg[x_col].to_numpy(), dfg[cost_col].to_numpy(), label=str(model_name), linewidth=1.2)
                    ax.set_title(f"Training cost curve ({label})")
                    ax.set_xlabel("epoch / eval index")
                    ax.set_ylabel("bill cost ($)")
                    ax.grid(True, alpha=0.2)
                    ax.legend(loc="best", fontsize="x-small")
                fig.suptitle("Early stopping: training cost evolution", fontsize=12)
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                log.info("Wrote %s", str(out_path))
    except Exception as e:
        log.info("fig_early_stopping_consistency generation skipped: %s", str(e))


def _slice_compiled(c: eil.CompiledTariff, sl: slice) -> eil.CompiledTariff:
    return eil.CompiledTariff(
        is_peak_window=np.asarray(c.is_peak_window[sl], dtype=int),
        energy_price=np.asarray(c.energy_price[sl], dtype=float),
        minutes_to_window_start=np.asarray(c.minutes_to_window_start[sl], dtype=int),
        minutes_to_window_end=np.asarray(c.minutes_to_window_end[sl], dtype=int),
        active_schedule_id=None if c.active_schedule_id is None else np.asarray(c.active_schedule_id[sl], dtype=int),
        is_super_offpeak=None if c.is_super_offpeak is None else np.asarray(c.is_super_offpeak[sl], dtype=int),
        is_midpeak=None if c.is_midpeak is None else np.asarray(c.is_midpeak[sl], dtype=int),
    )


def _resolve_feature_config(il_cfg: Mapping[str, Any], *, default_include_tariff: bool) -> eil.FeatureConfig:
    features_cfg = il_cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        return eil.FeatureConfig(include_tariff=default_include_tariff)

    active_name = str(features_cfg.get("active_feature_set", "")).strip()
    feature_sets = features_cfg.get("sets", {})
    if not active_name or not isinstance(feature_sets, dict) or active_name not in feature_sets:
        return eil.FeatureConfig(include_tariff=default_include_tariff)

    active_cfg = feature_sets.get(active_name, {})
    if not isinstance(active_cfg, dict):
        return eil.FeatureConfig(include_tariff=default_include_tariff)

    include_tariff = bool(default_include_tariff and (
        active_cfg.get("include_tariff_price", True)
        or active_cfg.get("include_peak_flag", True)
        or active_cfg.get("include_minutes_to_boundary", True)
    ))
    include_time = bool(active_cfg.get("include_calendar_index", True if not include_tariff else True))
    return eil.FeatureConfig(
        include_time=include_time,
        include_tariff=include_tariff,
        include_state=True,
        include_load=True,
        include_peak_flag=bool(active_cfg.get("include_peak_flag", True)),
        include_tariff_price=bool(active_cfg.get("include_tariff_price", True)),
        include_minutes_to_boundary=bool(active_cfg.get("include_minutes_to_boundary", True)),
    )


def _resolve_tariff_suite(*, il_cfg: Mapping[str, Any]) -> dict[str, eil.TariffIR]:
    experiment_cfg = il_cfg.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}

    active_families = experiment_cfg.get("active_tariff_families")
    if isinstance(active_families, list) and active_families:
        return eil.make_experiment_tariff_suite(
            timezone=str(il_cfg.get("timezone", "UTC")),
            weekmask=str(il_cfg.get("weekmask", "MON_FRI")),
        )

    tariffs_cfg = il_cfg.get("tariffs", {})
    if isinstance(tariffs_cfg, dict) and isinstance(tariffs_cfg.get("families"), dict):
        return eil.make_experiment_tariff_suite(
            timezone=str(il_cfg.get("timezone", "UTC")),
            weekmask=str(il_cfg.get("weekmask", "MON_FRI")),
        )

    return eil.make_paper_suite(
        timezone=str(il_cfg.get("timezone", "UTC")),
        weekmask=str(il_cfg.get("weekmask", "MON_FRI")),
    )


def _variant_labels_for_family(family_id: str, variants: list[int]) -> list[str]:
    prefix_map = {
        "peak_shift": "P",
        "seasonal_window": "S",
        "tiered_tou": "T",
    }
    if family_id not in prefix_map:
        raise ValueError(f"unknown family_id: {family_id}")
    prefix = prefix_map[family_id]
    return [f"{prefix}{int(v)}" for v in variants]


def _resolve_family_plans(*, il_cfg: Mapping[str, Any], suite: Mapping[str, eil.TariffIR]) -> list[dict[str, Any]]:
    experiment_cfg = il_cfg.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        experiment_cfg = {}
    training_cfg = il_cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        training_cfg = {}

    training_mode = str(experiment_cfg.get("training_mode", "global")).strip().lower()
    if training_mode != "family_specific":
        train_tariffs = il_cfg.get("train_tariffs", ["A", "B"])
        if not isinstance(train_tariffs, list) or not train_tariffs:
            train_tariffs = ["A", "B"]
        eval_tariffs = il_cfg.get("eval_tariffs", ["C", "C_flat"])
        if not isinstance(eval_tariffs, list) or not eval_tariffs:
            eval_tariffs = ["C", "C_flat"]
        return [{
            "family_id": None,
            "train_tariffs": [str(x) for x in train_tariffs],
            "eval_tariffs": [str(x) for x in eval_tariffs],
        }]

    active_families = experiment_cfg.get("active_tariff_families", ["peak_shift", "seasonal_window", "tiered_tou"])
    if not isinstance(active_families, list) or not active_families:
        active_families = ["peak_shift", "seasonal_window", "tiered_tou"]

    family_training_cfg = training_cfg.get("family_specific", {})
    if not isinstance(family_training_cfg, dict):
        family_training_cfg = {}
    target_families = family_training_cfg.get("target_families", active_families)
    if not isinstance(target_families, list) or not target_families:
        target_families = active_families

    tariffs_cfg = il_cfg.get("tariffs", {})
    families_cfg = tariffs_cfg.get("families", {}) if isinstance(tariffs_cfg, dict) else {}
    plans: list[dict[str, Any]] = []
    for family_id in [str(x) for x in target_families if str(x) in set(str(a) for a in active_families)]:
        family_cfg = families_cfg.get(family_id, {}) if isinstance(families_cfg, dict) else {}
        variants_cfg = family_cfg.get("variants", {}) if isinstance(family_cfg, dict) else {}
        train_variants = variants_cfg.get("train", [1, 2]) if isinstance(variants_cfg, dict) else [1, 2]
        held_out_variants = variants_cfg.get("held_out", [3, 4]) if isinstance(variants_cfg, dict) else [3, 4]
        train_tariffs = [t for t in _variant_labels_for_family(family_id, list(train_variants)) if t in suite]
        eval_tariffs = [t for t in _variant_labels_for_family(family_id, list(held_out_variants)) if t in suite]
        flat_variant = family_cfg.get("flat_confound_variant") if isinstance(family_cfg, dict) else None
        if family_id == "peak_shift" and flat_variant is not None:
            flat_label = f"P{int(flat_variant)}_flat"
            if flat_label in suite and flat_label not in eval_tariffs:
                eval_tariffs.append(flat_label)
        plans.append({
            "family_id": family_id,
            "train_tariffs": train_tariffs,
            "eval_tariffs": eval_tariffs,
        })
    return plans


def _resolve_forecast_variants(il_cfg: Mapping[str, Any], *, seed: int) -> dict[str, eil.ForecastNoiseConfig]:
    variants: dict[str, eil.ForecastNoiseConfig] = {
        "teacher_perfect": eil.ForecastNoiseConfig(sigma_rel=0.0, rho=0.0, seed=int(seed))
    }
    raw = il_cfg.get("forecast_robustness", {})
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return variants
    raw_variants = raw.get("variants", {})
    if not isinstance(raw_variants, dict):
        return variants
    for name, vcfg in raw_variants.items():
        if not isinstance(vcfg, dict):
            continue
        variants[str(name)] = eil.ForecastNoiseConfig(
            sigma_rel=float(vcfg.get("sigma_rel", 0.0)),
            rho=float(vcfg.get("rho", 0.0)),
            seed=int(vcfg.get("seed", seed)),
        )
    return variants


def _normalize_building_entries(il_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = il_cfg.get("building_manifest")
    if isinstance(manifest, list) and manifest:
        out: list[dict[str, Any]] = []
        for item in manifest:
            if not isinstance(item, dict) or item.get("building_id") is None:
                continue
            out.append({
                "building_id": str(item.get("building_id")),
                "timeseries_group": item.get("timeseries_group"),
                "archetype": item.get("archetype"),
                "group": item.get("timeseries_group"),
            })
        if out:
            return out
    building_ids = il_cfg.get("building_ids")
    if isinstance(building_ids, (list, tuple)) and building_ids:
        return [{"building_id": str(x)} for x in building_ids]
    if il_cfg.get("building_id") is not None:
        return [{"building_id": str(il_cfg.get("building_id"))}]
    return []


def _aggregate_multi_building_results(*, run_dir: Path) -> None:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    building_dirs = sorted(p for p in run_dir.glob("building_*") if p.is_dir())
    per_building_rows: list[pd.DataFrame] = []
    per_building_monthly_rows: list[pd.DataFrame] = []

    for building_dir in building_dirs:
        building_id = str(building_dir.name.replace("building_", "", 1))
        cfg_path = building_dir / "config_snapshot.json"
        archetype = None
        group = None
        if cfg_path.exists():
            try:
                snap = json.loads(cfg_path.read_text(encoding="utf-8"))
                il_snap = snap.get("il", {}) if isinstance(snap, dict) else {}
                if isinstance(il_snap, dict):
                    archetype = il_snap.get("archetype")
                    group = il_snap.get("timeseries_group")
            except Exception:
                pass

        # Collect metrics from experiment subdirectories
        exp_parent = building_dir / "experiments"
        if exp_parent.is_dir():
            exp_dirs = sorted(exp_parent.iterdir())
        else:
            # Fallback: treat building_dir itself as the single experiment
            exp_dirs = [building_dir]

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue
            experiment_name = exp_dir.name if exp_dir != building_dir else ""

            metrics_path = exp_dir / "metrics" / "metrics.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if not df.empty:
                    df.insert(0, "building_id", building_id)
                    if experiment_name:
                        df["experiment"] = experiment_name
                    df["archetype"] = archetype
                    df["timeseries_group"] = group
                    per_building_rows.append(df)

            monthly_path = exp_dir / "metrics" / "metrics_monthly.csv"
            if monthly_path.exists():
                dfm = pd.read_csv(monthly_path)
                if not dfm.empty:
                    dfm.insert(0, "building_id", building_id)
                    if experiment_name:
                        dfm["experiment"] = experiment_name
                    dfm["archetype"] = archetype
                    dfm["timeseries_group"] = group
                    per_building_monthly_rows.append(dfm)

    if not per_building_rows:
        return

    per_building = pd.concat(per_building_rows, axis=0, ignore_index=True)
    per_building.to_csv(aggregate_dir / "per_building_metrics.csv", index=False)

    group_cols = [col for col in ["experiment", "family", "tariff", "model", "forecast_variant"] if col in per_building.columns]
    agg = (
        per_building.groupby(group_cols, dropna=False, sort=False)
        .agg(
            n_buildings=("building_id", "nunique"),
            bill_total_mean=("bill_total", "mean"),
            bill_total_std=("bill_total", "std"),
            energy_cost_mean=("energy_cost", "mean"),
            energy_cost_std=("energy_cost", "std"),
            demand_cost_mean=("demand_cost", "mean"),
            demand_cost_std=("demand_cost", "std"),
            peak_kw_mean=("peak_kw", "mean"),
            peak_kw_std=("peak_kw", "std"),
            savings_vs_no_battery_mean=("savings_vs_no_battery", "mean"),
            savings_vs_no_battery_std=("savings_vs_no_battery", "std"),
            optimality_gap_vs_teacher_mean=("optimality_gap_vs_teacher", "mean"),
            optimality_gap_vs_teacher_std=("optimality_gap_vs_teacher", "std"),
        )
        .reset_index()
    )
    agg.to_csv(aggregate_dir / "aggregate_metrics_by_tariff_model.csv", index=False)

    if "family" in per_building.columns:
        # Exclude confound arms (e.g. P3_flat) from family-level aggregates
        # so they don't skew family means; they remain in per-tariff data.
        _confound_tariffs = {"P3_flat"}
        per_building_fam = per_building[~per_building["tariff"].isin(_confound_tariffs)]
        fam_group_cols = [col for col in ["experiment", "family", "model", "forecast_variant"] if col in per_building_fam.columns]
        fam = (
            per_building_fam.groupby(fam_group_cols, dropna=False, sort=False)
            .agg(
                n_buildings=("building_id", "nunique"),
                bill_total_mean=("bill_total", "mean"),
                bill_total_std=("bill_total", "std"),
                savings_vs_no_battery_mean=("savings_vs_no_battery", "mean"),
                savings_vs_no_battery_std=("savings_vs_no_battery", "std"),
                optimality_gap_vs_teacher_mean=("optimality_gap_vs_teacher", "mean"),
                optimality_gap_vs_teacher_std=("optimality_gap_vs_teacher", "std"),
            )
            .reset_index()
        )
        fam.to_csv(aggregate_dir / "aggregate_metrics_by_family.csv", index=False)

    if "archetype" in per_building.columns:
        arch_group = [col for col in ["experiment", "archetype", "model", "tariff"] if col in per_building.columns]
        arch = (
            per_building.groupby(arch_group, dropna=False, sort=False)
            .agg(
                n_buildings=("building_id", "nunique"),
                bill_total_mean=("bill_total", "mean"),
                bill_total_std=("bill_total", "std"),
                savings_vs_no_battery_mean=("savings_vs_no_battery", "mean"),
                savings_vs_no_battery_std=("savings_vs_no_battery", "std"),
            )
            .reset_index()
        )
        arch.to_csv(aggregate_dir / "aggregate_metrics_by_archetype.csv", index=False)

    if per_building_monthly_rows:
        monthly = pd.concat(per_building_monthly_rows, axis=0, ignore_index=True)
        monthly.to_csv(aggregate_dir / "per_building_monthly_metrics.csv", index=False)


def _run_family_plan(
    *,
    cfg: Mapping[str, Any],
    il_cfg: dict[str, Any],
    run_dir: Path,
    tracker: PipelineTracker,
    train_tariffs: list[str],
    eval_tariffs: list[str],
    family_suffix: str,
    plan_family_id: str | None,
    compiled: dict[str, tuple[eil.TariffIR, eil.CompiledTariff]],
    ts_utc: np.ndarray,
    load_kw: np.ndarray,
    tf: np.ndarray,
    battery: eil.BatteryParams,
    teacher: eil.TeacherConfig,
    horizon_steps: int,
    month_slices: list[tuple[int, slice]],
    train_months: set[int],
    val_months: set[int],
    test_months: set[int],
    allow_export: bool,
    init_E: float,
    auto_size: bool,
    window_hours: float,
    loaded: Any,
    T_use: int,
    split: Any,
    scales: eil.FeatureScales,
    train_idx: np.ndarray,
    feature_cfg_base: eil.FeatureConfig,
    feature_cfg_ta: eil.FeatureConfig,
    forecast_variants: dict[str, Any],
    suite_tz: str,
    run_teacher: bool = True,
    run_train: bool = True,
    run_eval: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run teacher generation, BC training, DAgger, and evaluation for one family plan.

    Returns (rows, runtime_rows, monthly_rows).
    """
    log = logging.getLogger("tail")
    family_label = str(plan_family_id) if plan_family_id else "default"

    teacher_dir = run_dir / ("teacher" + family_suffix)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    tracker.start_step(
        "il_teacher_generate" + family_suffix,
        tariffs=train_tariffs,
        family=family_label,
        horizon_steps=int(horizon_steps),
        battery_E_max_kwh=float(battery.E_max_kwh),
        battery_P_max_kw=float(battery.P_max_kw),
        battery_init_E_kwh=float(init_E),
        battery_auto_size=bool(auto_size),
        battery_sizing_window_hours=float(window_hours),
    )

    X_train_base_list: list[np.ndarray] = []
    X_val_base_list: list[np.ndarray] = []
    X_train_ta_list: list[np.ndarray] = []
    X_val_ta_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    y_val_list: list[np.ndarray] = []

    for name in train_tariffs:
        if name not in compiled:
            raise ValueError(f"unknown tariff in train_tariffs: {name}")
        tariff, c = compiled[name]

        # Resume / skip_if_outputs_exist: check for cached teacher rollout
        cached_npz = teacher_dir / name / "teacher_rollout.npz"
        if cached_npz.exists():
            log.info("IL teacher CACHED | family=%s | tariff=%s | loading %s", family_label, name, str(cached_npz))
            z = np.load(cached_npz, allow_pickle=True)
            _pbatt = np.asarray(z["P_batt_kw"], dtype=float)
            _grid_kw = np.asarray(z["grid_kw"], dtype=float) if "grid_kw" in z.files else _pbatt * 0.0
            rollout = eil.TeacherRollout(
                P_batt_kw=_pbatt,
                E_kwh=np.asarray(z["E_kwh"], dtype=float),
                current_max_peak_kw=np.asarray(z["current_max_peak_kw"], dtype=float),
                grid_kw=_grid_kw,
                grid_kw_clamped=np.asarray(z["grid_kw_clamped"], dtype=float),
                monthly_peak_kw_end=float(np.asarray(z["current_max_peak_kw"], dtype=float)[-1])
                    if np.asarray(z["current_max_peak_kw"]).size else 0.0,
            )
        else:
            if not run_teacher:
                raise RuntimeError(
                    f"run_teacher=false but no cached teacher rollout for family={family_label} tariff={name}. "
                    "Run teacher stage first or enable run_teacher."
                )
            P_hist = np.zeros(T_use, dtype=float)
            E_hist = np.zeros(T_use + 1, dtype=float)
            peak_hist = np.zeros(T_use + 1, dtype=float)
            grid_hist = np.zeros(T_use, dtype=float)
            gridc_hist = np.zeros(T_use, dtype=float)
            solves: list[eil.MPCResult] = []

            def _solve_month(sl: slice) -> tuple[slice, eil.TeacherRollout, list[eil.MPCResult]]:
                r, s = eil.run_teacher_receding_horizon(
                    load_kw=load_kw[sl],
                    energy_price_per_kwh=c.energy_price[sl],
                    is_peak_window=c.is_peak_window[sl],
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    battery=battery,
                    allow_grid_export=allow_export,
                    teacher=teacher,
                    horizon_steps=horizon_steps,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                )
                return sl, r, s

            with cf.ThreadPoolExecutor(max_workers=len(month_slices)) as pool:
                for sl, rollout_m, solves_m in pool.map(
                    lambda ms: _solve_month(ms[1]), month_slices
                ):
                    P_hist[sl] = np.asarray(rollout_m.P_batt_kw, dtype=float)
                    E_hist[sl.start : sl.stop + 1] = np.asarray(rollout_m.E_kwh, dtype=float)
                    peak_hist[sl.start : sl.stop + 1] = np.asarray(rollout_m.current_max_peak_kw, dtype=float)
                    grid_hist[sl] = np.asarray(rollout_m.grid_kw, dtype=float)
                    gridc_hist[sl] = np.asarray(rollout_m.grid_kw_clamped, dtype=float)
                    solves.extend(solves_m)

            rollout = eil.TeacherRollout(
                P_batt_kw=P_hist,
                E_kwh=E_hist,
                current_max_peak_kw=peak_hist,
                grid_kw=grid_hist,
                grid_kw_clamped=gridc_hist,
                monthly_peak_kw_end=float(peak_hist[-1]) if peak_hist.size else 0.0,
            )

            (teacher_dir / name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                teacher_dir / name / "teacher_rollout.npz",
                P_batt_kw=rollout.P_batt_kw,
                E_kwh=rollout.E_kwh,
                current_max_peak_kw=rollout.current_max_peak_kw,
                grid_kw=rollout.grid_kw,
                grid_kw_clamped=rollout.grid_kw_clamped,
            )

            _soc_kwh_teacher = np.asarray(rollout.E_kwh[:-1], dtype=float)
            df_teacher = pd.DataFrame(
                {
                    "timestamp": ts_utc.astype(str),
                    "load_kw": np.asarray(load_kw, dtype=float),
                    "soc_kwh": _soc_kwh_teacher,
                    "soc_pct": _soc_kwh_teacher / battery.E_max_kwh * 100.0,
                    "p_batt_star_kw": np.asarray(rollout.P_batt_kw, dtype=float),
                    "tariff_id": str(name),
                    "is_peak_window": np.asarray(c.is_peak_window, dtype=int),
                    "energy_price": np.asarray(c.energy_price, dtype=float),
                    "minutes_to_window_start": np.asarray(c.minutes_to_window_start, dtype=float),
                    "minutes_to_window_end": np.asarray(c.minutes_to_window_end, dtype=float),
                    "current_max_peak_kw": np.asarray(rollout.current_max_peak_kw[:-1], dtype=float),
                }
            )
            df_teacher.to_csv(teacher_dir / name / "teacher_rollout.csv", index=False)

            write_json(
                teacher_dir / name / "solver_summary.json",
                {
                    "requested_solver": teacher.solver,
                    "sample_solver_used_first_200": [s.solver_log.solver_used for s in solves[: min(len(solves), 200)]],
                    "used_fallback_any": bool(any(s.solver_log.used_fallback_solver for s in solves)),
                },
            )

        X_all_ta = eil.make_features(
            load_kw=load_kw,
            state_E_kwh=np.asarray(rollout.E_kwh[:-1], dtype=float),
            state_current_max_peak_kw=np.asarray(rollout.current_max_peak_kw[:-1], dtype=float),
            compiled=c,
            time_features=tf,
            cfg=feature_cfg_ta,
            battery_E_max_kwh=float(battery.E_max_kwh),
            scales=scales,
        )
        X_all_base = eil.make_features(
            load_kw=load_kw,
            state_E_kwh=np.asarray(rollout.E_kwh[:-1], dtype=float),
            state_current_max_peak_kw=np.asarray(rollout.current_max_peak_kw[:-1], dtype=float),
            compiled=c,
            time_features=tf,
            cfg=feature_cfg_base,
            battery_E_max_kwh=float(battery.E_max_kwh),
            scales=scales,
        )

        y_all = np.asarray(rollout.P_batt_kw, dtype=np.float32)

        X_train_ta_list.append(X_all_ta[train_idx])
        X_train_base_list.append(X_all_base[train_idx])
        y_train_list.append(y_all[train_idx])

        if split.val_idx.size:
            X_val_ta_list.append(X_all_ta[split.val_idx])
            X_val_base_list.append(X_all_base[split.val_idx])
            y_val_list.append(y_all[split.val_idx])

        log.info("IL teacher done | family=%s | tariff=%s | T=%d", family_label, name, int(len(rollout.P_batt_kw)))

    X_train_ta = np.concatenate(X_train_ta_list, axis=0)
    X_train_base = np.concatenate(X_train_base_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    if X_val_ta_list:
        X_val_ta = np.concatenate(X_val_ta_list, axis=0)
        X_val_base = np.concatenate(X_val_base_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
    else:
        X_val_ta = np.zeros((0, int(X_train_ta.shape[1])), dtype=X_train_ta.dtype)
        X_val_base = np.zeros((0, int(X_train_base.shape[1])), dtype=X_train_base.dtype)
        y_val = np.zeros((0,), dtype=y_train.dtype)

    if X_val_ta.shape[0] == 0:
        n = int(X_train_ta.shape[0])
        cut = max(1, int(0.9 * n))
        X_val_ta = X_train_ta[cut:]
        X_val_base = X_train_base[cut:]
        y_val = y_train[cut:]
        X_train_ta = X_train_ta[:cut]
        X_train_base = X_train_base[:cut]
        y_train = y_train[:cut]

    tracker.end_step("il_teacher_generate" + family_suffix, ok=True, n_train=int(X_train_ta.shape[0]), n_val=int(X_val_ta.shape[0]))

    if torch is None:
        raise RuntimeError("torch is required for student training")

    train_dir = run_dir / ("models" + family_suffix)
    train_dir.mkdir(parents=True, exist_ok=True)

    bc_cfg_raw = il_cfg.get("bc", {})
    if not isinstance(bc_cfg_raw, dict):
        bc_cfg_raw = {}
    cost_eval_every = int(bc_cfg_raw.get("cost_eval_every", 25))
    cost_patience_evals = int(bc_cfg_raw.get("cost_patience_evals", 2))
    train_cfg = eil.TrainConfig(
        epochs=min(int(bc_cfg_raw.get("epochs", 10)), 200),
        batch_size=int(bc_cfg_raw.get("batch_size", 256)),
        lr=float(bc_cfg_raw.get("lr", 3e-4)),
        seed=int(cfg.get("seed", 1337)),
        weight_decay=float(bc_cfg_raw.get("weight_decay", 0.0)),
        early_stopping_patience=0,
        early_stopping_min_delta=float(bc_cfg_raw.get("early_stopping_min_delta", 0.0)),
        cost_eval_every=int(cost_eval_every),
        cost_patience_evals=int(cost_patience_evals),
        device=str(bc_cfg_raw.get("device", "cpu")),
        restore_best_checkpoint_by=str(bc_cfg_raw.get("restore_best_checkpoint_by", "bill_cost")),
    )

    def _eval_cost_on_val_split(
        *,
        model: Any,
        feature_cfg: eil.FeatureConfig,
        model_name: str,
    ) -> dict[str, Any]:
        if not val_months:
            return {
                "model": model_name,
                "cost_total": float("nan"),
                "energy_cost": float("nan"),
                "demand_cost": float("nan"),
                "peak_kw": float("nan"),
                "months": 0,
                "tariffs": list(train_tariffs),
            }

        eval_month_slices = [(m, sl) for (m, sl) in month_slices if int(m) in val_months]
        if not eval_month_slices:
            return {
                "model": model_name,
                "cost_total": float("nan"),
                "energy_cost": float("nan"),
                "demand_cost": float("nan"),
                "peak_kw": float("nan"),
                "months": 0,
                "tariffs": list(train_tariffs),
            }

        bill_total = 0.0
        energy_cost = 0.0
        demand_cost = 0.0
        peak_kw = 0.0
        months_count = 0

        for _, sl in eval_month_slices:
            for tname in train_tariffs:
                if tname not in compiled:
                    continue
                tariff_v, c_full = compiled[tname]
                load_seg = load_kw[sl]
                tf_seg = tf[sl]
                c_seg = _slice_compiled(c_full, sl)

                rr = eil.rollout_policy(
                    model=model,
                    load_kw=load_seg,
                    compiled=c_seg,
                    time_features=tf_seg,
                    battery=battery,
                    allow_grid_export=allow_export,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    feature_cfg=feature_cfg,
                    scales=scales,
                )
                bill = eil.compute_monthly_bill(
                    grid_kw=rr.grid_kw_clamped,
                    energy_price_per_kwh=c_seg.energy_price,
                    is_peak_window=c_seg.is_peak_window,
                    demand_charge_rate_kw=float(tariff_v.demand_charge_rate_kw),
                    dt_hours=float(loaded.dt_hours),
                )
                bill_total += float(bill.bill_total)
                energy_cost += float(bill.energy_cost)
                demand_cost += float(bill.demand_cost)
                peak_kw = max(float(peak_kw), float(bill.peak_kw))
            months_count += 1

        return {
            "model": model_name,
            "split": "val",
            "cost_total": float(bill_total),
            "energy_cost": float(energy_cost),
            "demand_cost": float(demand_cost),
            "peak_kw": float(peak_kw),
            "months": int(months_count),
            "tariffs": list(train_tariffs),
            "val_months": sorted(list(val_months)),
        }

    tracker.start_step("il_train_bc" + family_suffix)

    # skip_if_outputs_exist: check for existing trained models
    _bc_base_pt = train_dir / "bc_baseline.pt"
    _bc_ta_pt = train_dir / "bc_tariff_aware.pt"
    _skip_train = bool(not run_train and _bc_base_pt.exists() and _bc_ta_pt.exists())

    if _skip_train:
        log.info("SKIP training (run_train=false, cached models found) | family=%s", family_label)
        _ckpt_base = torch.load(_bc_base_pt, map_location="cpu", weights_only=True)
        model_base = eil.MLPPolicy(input_dim=int(_ckpt_base["input_dim"]), cfg=eil.MLPConfig())
        model_base.load_state_dict(_ckpt_base["state_dict"])
        _ckpt_ta = torch.load(_bc_ta_pt, map_location="cpu", weights_only=True)
        model_ta = eil.MLPPolicy(input_dim=int(_ckpt_ta["input_dim"]), cfg=eil.MLPConfig())
        model_ta.load_state_dict(_ckpt_ta["state_dict"])
        res_base = res_ta = None  # no training results available
        tracker.end_step("il_train_bc" + family_suffix, ok=True, skipped=True)
    else:
        res_base = eil.fit_bc_policy(
            X=X_train_base,
            y_action_kw=y_train,
            train=train_cfg,
            model_cfg=eil.MLPConfig(),
            X_val=X_val_base if X_val_base.size else None,
            y_val=y_val if y_val.size else None,
            cost_eval_fn=lambda m: _eval_cost_on_val_split(model=m, feature_cfg=feature_cfg_base, model_name="bc_baseline"),
        )
        torch.save({"state_dict": res_base.model.state_dict(), "input_dim": int(X_train_base.shape[1])}, train_dir / "bc_baseline.pt")

        res_ta = eil.fit_bc_policy(
            X=X_train_ta,
            y_action_kw=y_train,
            train=train_cfg,
            model_cfg=eil.MLPConfig(),
            X_val=X_val_ta if X_val_ta.size else None,
            y_val=y_val if y_val.size else None,
            cost_eval_fn=lambda m: _eval_cost_on_val_split(model=m, feature_cfg=feature_cfg_ta, model_name="bc_tariff_aware"),
        )
        torch.save({"state_dict": res_ta.model.state_dict(), "input_dim": int(X_train_ta.shape[1])}, train_dir / "bc_tariff_aware.pt")

        # Save MSE-selected checkpoint for early-stopping diagnostic
        if res_ta.best_val_mse_state is not None:
            torch.save(
                {"state_dict": res_ta.best_val_mse_state, "input_dim": int(X_train_ta.shape[1])},
                train_dir / "bc_tariff_aware_mse_selected.pt",
            )

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        losses_csv = metrics_dir / ("train_losses" + family_suffix + ".csv")
        if not losses_csv.exists():
            rows_loss: list[dict[str, Any]] = []
            if res_base.loss_history:
                for i, v in enumerate(res_base.loss_history, start=1):
                    row: dict[str, Any] = {"family": family_label, "model": "bc_baseline", "epoch": int(i), "loss_mse": float(v)}
                    if res_base.val_loss_history and i <= len(res_base.val_loss_history):
                        row["val_loss_mse"] = float(res_base.val_loss_history[i - 1])
                    rows_loss.append(row)
            if res_ta.loss_history:
                for i, v in enumerate(res_ta.loss_history, start=1):
                    row = {"family": family_label, "model": "bc_tariff_aware", "epoch": int(i), "loss_mse": float(v)}
                    if res_ta.val_loss_history and i <= len(res_ta.val_loss_history):
                        row["val_loss_mse"] = float(res_ta.val_loss_history[i - 1])
                    rows_loss.append(row)
            if rows_loss:
                pd.DataFrame(rows_loss).to_csv(losses_csv, index=False)
                log.info("Wrote %s", str(losses_csv))

        cost_csv = metrics_dir / ("train_costs" + family_suffix + ".csv")
        if not cost_csv.exists():
            rows_cost: list[dict[str, Any]] = []
            for res in (res_base, res_ta):
                if res.cost_eval_history:
                    for r in res.cost_eval_history:
                        rows_cost.append(dict(r))
            if rows_cost:
                pd.DataFrame(rows_cost).to_csv(cost_csv, index=False)
                log.info("Wrote %s", str(cost_csv))

        # Checkpoint selection diagnostic: bill-selected vs MSE-selected
        diag_csv = metrics_dir / ("checkpoint_selection_diagnostic" + family_suffix + ".csv")
        if not diag_csv.exists() and res_ta.cost_eval_history:
            _diag_rows: list[dict[str, Any]] = []
            for entry in res_ta.cost_eval_history:
                _diag_rows.append({
                    "family": family_label,
                    "model": entry.get("model", "bc_tariff_aware"),
                    "epoch": entry.get("epoch"),
                    "val_bill_cost": entry.get("cost_total"),
                    "val_loss_mse": entry.get("val_loss_mse", float("nan")),
                })
            if _diag_rows:
                _df_diag = pd.DataFrame(_diag_rows)
                _bill_finite = _df_diag["val_bill_cost"].replace([float("inf"), float("-inf")], float("nan"))
                _mse_finite = _df_diag["val_loss_mse"].replace([float("inf"), float("-inf")], float("nan"))
                _df_diag["selected_by_bill"] = False
                _df_diag["selected_by_mse"] = False
                if _bill_finite.notna().any():
                    _df_diag.loc[_bill_finite.idxmin(), "selected_by_bill"] = True
                if _mse_finite.notna().any():
                    _df_diag.loc[_mse_finite.idxmin(), "selected_by_mse"] = True
                _df_diag.to_csv(diag_csv, index=False)
                log.info("Wrote %s", str(diag_csv))

        tracker.end_step("il_train_bc" + family_suffix, ok=True, train_loss_baseline=float(res_base.train_loss), train_loss_ta=float(res_ta.train_loss))

        # Write model architecture manifest
        def _model_param_count(m: Any) -> int:
            return sum(int(p.numel()) for p in m.parameters())

        model_manifest_path = train_dir / "model_manifest.json"
        if not model_manifest_path.exists():
            mlp_cfg = eil.MLPConfig()
            write_json(model_manifest_path, {
                "family": family_label,
                "hidden_sizes": list(mlp_cfg.hidden_sizes),
                "activation": "ReLU",
                "bc_baseline": {
                    "input_dim": int(X_train_base.shape[1]),
                    "param_count": _model_param_count(res_base.model),
                    "train_loss": float(res_base.train_loss),
                    "best_epoch": res_base.best_epoch,
                    "total_epochs": res_base.total_epochs,
                    "restored_by": res_base.restored_by,
                    "best_val_loss": res_base.best_val_loss,
                    "best_cost": res_base.best_cost,
                },
                "bc_tariff_aware": {
                    "input_dim": int(X_train_ta.shape[1]),
                    "param_count": _model_param_count(res_ta.model),
                    "train_loss": float(res_ta.train_loss),
                    "best_epoch": res_ta.best_epoch,
                    "total_epochs": res_ta.total_epochs,
                    "restored_by": res_ta.restored_by,
                    "best_val_loss": res_ta.best_val_loss,
                    "best_cost": res_ta.best_cost,
                    "best_val_mse_epoch": res_ta.best_val_mse_epoch,
                },
            })
            log.info("Wrote %s", str(model_manifest_path))

        model_base = res_base.model
        model_ta = res_ta.model

    # Load MSE-selected checkpoint (for early-stopping diagnostic eval)
    model_ta_mse_selected = None
    _bc_mse_pt = train_dir / "bc_tariff_aware_mse_selected.pt"
    if _bc_mse_pt.exists():
        _ckpt_mse = torch.load(_bc_mse_pt, map_location="cpu", weights_only=True)
        model_ta_mse_selected = eil.MLPPolicy(input_dim=int(_ckpt_mse["input_dim"]), cfg=eil.MLPConfig())
        model_ta_mse_selected.load_state_dict(_ckpt_mse["state_dict"])
        model_ta_mse_selected.eval()
        log.info("Loaded MSE-selected checkpoint for early-stopping diagnostic")

    model_base_dagger = None
    model_ta_dagger = None

    dag_raw = il_cfg.get("dagger", {})
    do_dagger = bool(dag_raw.get("enabled", False)) if isinstance(dag_raw, dict) else False

    if do_dagger:
        dag_beta = float(dag_raw.get("beta", 0.0))
        dcfg = eil.DAggerConfig(
            enabled=True,
            rollout_steps=int(dag_raw.get("rollout_steps", min(2000, T_use))),
            label_stride=int(dag_raw.get("label_stride", 1)),
            mpc_horizon_steps=int(dag_raw.get("mpc_horizon_steps", horizon_steps)),
            beta=dag_beta,
        )
        # Use first train tariff for DAgger relabeling (was hardcoded to "A")
        dag_tariff_name = train_tariffs[0]
        tariffD, cD = compiled[dag_tariff_name]
        dag_rounds = max(1, int(dag_raw.get("rounds", 1)))
        dag_retrain_scratch = bool(dag_raw.get("retrain_from_scratch", False))

        tracker.start_step("il_dagger" + family_suffix)

        def _collect_dagger_over_months(
            *,
            model: Any,
            feature_cfg: eil.FeatureConfig,
        ) -> tuple[np.ndarray, np.ndarray, eil.DAggerRoundStats, dict[str, np.ndarray]]:
            X_list: list[np.ndarray] = []
            y_list: list[np.ndarray] = []
            trace_all: dict[str, list[Any]] = {}
            n_labeled = 0

            for month, sl in month_slices:
                if train_months and (int(month) not in train_months):
                    continue
                c_seg = _slice_compiled(cD, sl)
                X_m, y_m, s_m, trace_m = eil.dagger_collect_labels_with_trace(
                    model=model,
                    load_kw=load_kw[sl],
                    compiled=c_seg,
                    time_features=tf[sl],
                    battery=battery,
                    allow_grid_export=allow_export,
                    teacher=teacher,
                    demand_charge_rate_kw=float(tariffD.demand_charge_rate_kw),
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    feature_cfg=feature_cfg,
                    scales=scales,
                    cfg=dcfg,
                    timestamps_utc=ts_utc[sl],
                )
                if X_m.size:
                    X_list.append(X_m)
                if y_m.size:
                    y_list.append(y_m)
                n_labeled += int(s_m.n_labeled)

                if "t" in trace_m:
                    trace_m["t"] = np.asarray(trace_m["t"], dtype=int) + int(sl.start)

                for k, v in trace_m.items():
                    trace_all.setdefault(k, []).extend(list(v))

            if X_list:
                X_out = np.concatenate(X_list, axis=0)
            else:
                X_out = np.zeros((0, 0), dtype=np.float32)
            if y_list:
                y_out = np.concatenate(y_list, axis=0)
            else:
                y_out = np.zeros((0,), dtype=np.float32)

            trace_out = {k: np.asarray(v) for k, v in trace_all.items()}
            return X_out, y_out, eil.DAggerRoundStats(n_labeled=n_labeled), trace_out

        dagger_dir = teacher_dir / "dagger"
        dagger_dir.mkdir(parents=True, exist_ok=True)

        # ── beta=1.0 teacher‐cache reuse optimisation ──────────────────────
        # When beta>=1.0 the teacher steers every step, so the resulting
        # trajectory is identical to the cached teacher rollout.  We extract
        # (state, action) pairs directly instead of re-solving MPC.
        def _extract_dagger_from_teacher_cache(
            *,
            feature_cfg: eil.FeatureConfig,
        ) -> tuple[np.ndarray, np.ndarray, eil.DAggerRoundStats, dict[str, np.ndarray]]:
            cached_npz = teacher_dir / dag_tariff_name / "teacher_rollout.npz"
            z = np.load(cached_npz, allow_pickle=True)
            P_batt = np.asarray(z["P_batt_kw"], dtype=float)
            E = np.asarray(z["E_kwh"], dtype=float)
            peak = np.asarray(z["current_max_peak_kw"], dtype=float)

            X_list_c: list[np.ndarray] = []
            y_list_c: list[np.ndarray] = []
            for month, sl in month_slices:
                if train_months and (int(month) not in train_months):
                    continue
                c_seg = _slice_compiled(cD, sl)
                X_seg = eil.make_features(
                    load_kw=load_kw[sl],
                    state_E_kwh=E[sl.start : sl.stop],
                    state_current_max_peak_kw=peak[sl.start : sl.stop],
                    compiled=c_seg,
                    time_features=tf[sl],
                    cfg=feature_cfg,
                    battery_E_max_kwh=float(battery.E_max_kwh),
                    scales=scales,
                )
                X_list_c.append(X_seg)
                y_list_c.append(P_batt[sl].astype(np.float32))

            X_out_c = np.concatenate(X_list_c, axis=0) if X_list_c else np.zeros((0, 0), dtype=np.float32)
            y_out_c = np.concatenate(y_list_c, axis=0) if y_list_c else np.zeros((0,), dtype=np.float32)
            trace_c: dict[str, np.ndarray] = {"source": np.array(["teacher_cache"] * len(y_out_c), dtype=object)}
            return X_out_c, y_out_c, eil.DAggerRoundStats(n_labeled=int(y_out_c.shape[0])), trace_c

        use_teacher_cache = bool(dag_beta >= 1.0)
        if use_teacher_cache:
            log.info("DAgger beta>=1.0 → reusing cached teacher rollout (no MPC re-solves)")

        # Aggregated datasets start from BC training data
        Xb_agg = X_train_base.copy()
        yb_agg = y_train.copy()
        Xt_agg = X_train_ta.copy()
        yt_agg = y_train.copy()

        cur_model_base = model_base
        cur_model_ta = model_ta
        round_summary_rows: list[dict[str, Any]] = []
        rows_loss_dagger: list[dict[str, Any]] = []

        for R in range(dag_rounds):
            round_dir = dagger_dir / f"round_{R}"
            round_dir.mkdir(parents=True, exist_ok=True)

            if use_teacher_cache:
                # beta>=1.0: reuse teacher rollout (no MPC re-solves needed)
                X_new_b, y_new_b, s_b, trace_b = _extract_dagger_from_teacher_cache(
                    feature_cfg=feature_cfg_base,
                )
                X_new_ta, y_new_ta, s_ta, trace_ta = _extract_dagger_from_teacher_cache(
                    feature_cfg=feature_cfg_ta,
                )
            else:
                X_new_b, y_new_b, s_b, trace_b = _collect_dagger_over_months(
                    model=cur_model_base,
                    feature_cfg=feature_cfg_base,
                )
                X_new_ta, y_new_ta, s_ta, trace_ta = _collect_dagger_over_months(
                    model=cur_model_ta,
                    feature_cfg=feature_cfg_ta,
                )

            out_dag_b = round_dir / f"labels_baseline_{dag_tariff_name}.csv"
            if not out_dag_b.exists():
                pd.DataFrame({k: v for k, v in trace_b.items()}).to_csv(out_dag_b, index=False)
                log.info("Wrote %s", str(out_dag_b))

            out_dag_ta = round_dir / f"labels_tariff_aware_{dag_tariff_name}.csv"
            if not out_dag_ta.exists():
                pd.DataFrame({k: v for k, v in trace_ta.items()}).to_csv(out_dag_ta, index=False)
                log.info("Wrote %s", str(out_dag_ta))

            # Aggregate new labels with all prior data
            if X_new_b.size:
                Xb_agg = np.concatenate([Xb_agg, X_new_b], axis=0)
                yb_agg = np.concatenate([yb_agg, y_new_b], axis=0)
            if X_new_ta.size:
                Xt_agg = np.concatenate([Xt_agg, X_new_ta], axis=0)
                yt_agg = np.concatenate([yt_agg, y_new_ta], axis=0)

            # Retrain on aggregated data
            round_label = f"dagger_baseline_r{R}"
            res_bd = eil.fit_bc_policy(
                X=X_train_base if dag_retrain_scratch else Xb_agg,
                y_action_kw=y_train if dag_retrain_scratch else yb_agg,
                train=train_cfg,
                model_cfg=eil.MLPConfig(),
                cost_eval_fn=lambda m, _rl=round_label: _eval_cost_on_val_split(model=m, feature_cfg=feature_cfg_base, model_name=_rl),
            )
            cur_model_base = res_bd.model
            torch.save(
                {"state_dict": cur_model_base.state_dict(), "input_dim": int(Xb_agg.shape[1])},
                round_dir / "dagger_baseline.pt",
            )

            round_label_ta = f"dagger_tariff_aware_r{R}"
            res_td = eil.fit_bc_policy(
                X=X_train_ta if dag_retrain_scratch else Xt_agg,
                y_action_kw=y_train if dag_retrain_scratch else yt_agg,
                train=train_cfg,
                model_cfg=eil.MLPConfig(),
                cost_eval_fn=lambda m, _rl=round_label_ta: _eval_cost_on_val_split(model=m, feature_cfg=feature_cfg_ta, model_name=_rl),
            )
            cur_model_ta = res_td.model
            torch.save(
                {"state_dict": cur_model_ta.state_dict(), "input_dim": int(Xt_agg.shape[1])},
                round_dir / "dagger_tariff_aware.pt",
            )

            # Per-round losses
            if res_bd.loss_history:
                for i, v in enumerate(res_bd.loss_history, start=1):
                    row: dict[str, Any] = {"family": family_label, "model": f"dagger_baseline_r{R}", "epoch": int(i), "loss_mse": float(v)}
                    if res_bd.val_loss_history and i <= len(res_bd.val_loss_history):
                        row["val_loss_mse"] = float(res_bd.val_loss_history[i - 1])
                    rows_loss_dagger.append(row)
            if res_td.loss_history:
                for i, v in enumerate(res_td.loss_history, start=1):
                    row = {"family": family_label, "model": f"dagger_tariff_aware_r{R}", "epoch": int(i), "loss_mse": float(v)}
                    if res_td.val_loss_history and i <= len(res_td.val_loss_history):
                        row["val_loss_mse"] = float(res_td.val_loss_history[i - 1])
                    rows_loss_dagger.append(row)

            round_summary_rows.append({
                "family": family_label,
                "round": int(R),
                "n_labeled_baseline": int(s_b.n_labeled),
                "n_labeled_ta": int(s_ta.n_labeled),
                "n_agg_baseline": int(Xb_agg.shape[0]),
                "n_agg_ta": int(Xt_agg.shape[0]),
                "train_loss_baseline": float(res_bd.train_loss),
                "train_loss_ta": float(res_td.train_loss),
                "best_epoch_baseline": res_bd.best_epoch,
                "best_epoch_ta": res_td.best_epoch,
                "total_epochs_baseline": res_bd.total_epochs,
                "total_epochs_ta": res_td.total_epochs,
                "restored_by_baseline": res_bd.restored_by,
                "restored_by_ta": res_td.restored_by,
                "best_val_loss_baseline": res_bd.best_val_loss,
                "best_val_loss_ta": res_td.best_val_loss,
                "best_cost_baseline": res_bd.best_cost,
                "best_cost_ta": res_td.best_cost,
            })
            log.info(
                "DAgger round %d/%d done | family=%s | n_labeled_b=%d n_labeled_ta=%d | n_agg_b=%d n_agg_ta=%d",
                R, dag_rounds, family_label, s_b.n_labeled, s_ta.n_labeled,
                Xb_agg.shape[0], Xt_agg.shape[0],
            )

        model_base_dagger = cur_model_base
        model_ta_dagger = cur_model_ta

        # Save final models at canonical location
        torch.save({"state_dict": model_base_dagger.state_dict(), "input_dim": int(Xb_agg.shape[1])}, train_dir / "dagger_baseline.pt")
        torch.save({"state_dict": model_ta_dagger.state_dict(), "input_dim": int(Xt_agg.shape[1])}, train_dir / "dagger_tariff_aware.pt")

        # Write per-round losses
        if rows_loss_dagger:
            losses_csv_d = metrics_dir / ("train_losses_dagger" + family_suffix + ".csv")
            if not losses_csv_d.exists():
                pd.DataFrame(rows_loss_dagger).to_csv(losses_csv_d, index=False)
                log.info("Wrote %s", str(losses_csv_d))

        # Write per-round cost evals (across all rounds)
        rows_cost_d: list[dict[str, Any]] = []
        # (cost_eval_history is captured via the last-round results; for multi-round, each round's res was already used)
        for res in (res_bd, res_td):
            if res.cost_eval_history:
                for r in res.cost_eval_history:
                    rows_cost_d.append(dict(r))
        if rows_cost_d:
            cost_csv_d = metrics_dir / ("train_costs_dagger" + family_suffix + ".csv")
            if not cost_csv_d.exists():
                pd.DataFrame(rows_cost_d).to_csv(cost_csv_d, index=False)
                log.info("Wrote %s", str(cost_csv_d))

        # Write round summary
        if round_summary_rows:
            pd.DataFrame(round_summary_rows).to_csv(
                metrics_dir / ("dagger_round_summary" + family_suffix + ".csv"),
                index=False,
            )
            log.info("Wrote dagger_round_summary%s.csv", family_suffix)

        tracker.end_step(
            "il_dagger" + family_suffix,
            ok=True,
            rounds=int(dag_rounds),
            n_labeled_baseline_total=sum(r["n_labeled_baseline"] for r in round_summary_rows),
            n_labeled_ta_total=sum(r["n_labeled_ta"] for r in round_summary_rows),
        )

    # --- Evaluation ---
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    monthly_rows_all: list[dict[str, Any]] = []

    if not run_eval:
        log.info("SKIP eval (run_eval=false) | family=%s", family_label)
        tracker.start_step("il_eval" + family_suffix)
        tracker.end_step("il_eval" + family_suffix, ok=True, skipped=True)
        return rows, runtime_rows, monthly_rows_all

    # --- Eval resume: reload from cached CSVs if all eval NPZs already exist ---
    metrics_dir_check = run_dir / "metrics"
    _cached_metrics_csv = metrics_dir_check / "metrics.csv"
    _cached_monthly_csv = metrics_dir_check / "metrics_monthly.csv"
    _cached_runtimes_csv = metrics_dir_check / "runtimes.csv"
    _all_eval_npz_exist = all(
        (metrics_dir_check / f"eval_rollouts_{family_label}_{t}.npz").exists()
        for t in eval_tariffs
    )
    if _all_eval_npz_exist and _cached_metrics_csv.exists():
        try:
            _df_cached = pd.read_csv(_cached_metrics_csv)
            _family_mask = _df_cached["family"] == family_label if "family" in _df_cached.columns else pd.Series(True, index=_df_cached.index)
            _df_family = _df_cached[_family_mask]
            if not _df_family.empty:
                rows = _df_family.to_dict(orient="records")
                if _cached_monthly_csv.exists():
                    _df_m = pd.read_csv(_cached_monthly_csv)
                    _m_mask = _df_m["family"] == family_label if "family" in _df_m.columns else pd.Series(True, index=_df_m.index)
                    monthly_rows_all = _df_m[_m_mask].to_dict(orient="records")
                if _cached_runtimes_csv.exists():
                    _df_r = pd.read_csv(_cached_runtimes_csv)
                    _r_mask = _df_r["family"] == family_label if "family" in _df_r.columns else pd.Series(True, index=_df_r.index)
                    runtime_rows = _df_r[_r_mask].to_dict(orient="records")
                log.info("IL eval CACHED | family=%s | loaded %d rows from metrics.csv", family_label, len(rows))
                tracker.start_step("il_eval" + family_suffix)
                tracker.end_step("il_eval" + family_suffix, ok=True, cached=True, rows=int(len(rows)))
                return rows, runtime_rows, monthly_rows_all
        except Exception:
            log.debug("Eval cache reload failed for family=%s, running fresh eval", family_label)

    tracker.start_step("il_eval" + family_suffix)
    for tname in eval_tariffs:
        if tname not in compiled:
            raise ValueError(f"unknown eval tariff: {tname}")
        tariff, c_full = compiled[tname]

        eval_month_slices: list[tuple[int, slice]]
        if test_months:
            eval_month_slices = [(m, sl) for (m, sl) in month_slices if m in test_months]
        else:
            eval_month_slices = [(0, slice(0, T_use))]

        bills: dict[str, dict[str, float]] = {
            "no_battery": {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0},
            "bc_baseline": {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0},
            "bc_tariff_aware": {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0},
            "teacher": {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0},
        }
        if model_base_dagger is not None:
            bills["dagger_baseline"] = {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0}
        if model_ta_dagger is not None:
            bills["dagger_tariff_aware"] = {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0}
        if model_ta_mse_selected is not None:
            bills["bc_tariff_aware_mse_selected"] = {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0}

        # Rule baseline: threshold from train months on this eval tariff's peak windows
        rule_cfg = eil.compute_rule_baseline_config(
            load_kw=load_kw,
            compiled=c_full,
            timestamps_utc=ts_utc,
            timezone=suite_tz,
            train_months=sorted(train_months),
        )
        bills["rule_tariff_aware"] = {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0}

        # Persist rule baseline config for auditability
        _rule_artifact_dir = run_dir / "metrics"
        _rule_artifact_dir.mkdir(parents=True, exist_ok=True)
        _rule_json_path = _rule_artifact_dir / f"rule_baseline_config_{family_label}_{tname}.json"
        if not _rule_json_path.exists():
            try:
                write_json(_rule_json_path, {
                    "family": family_label,
                    "tariff": tname,
                    "theta_kw": float(rule_cfg.theta_kw),
                    "precharge_window_minutes": float(rule_cfg.precharge_window_minutes),
                    "train_months": sorted(list(train_months)),
                })
            except Exception:
                pass

        # Noisy teacher forecast variants
        noisy_variants: dict[str, eil.ForecastNoiseConfig] = {}
        if isinstance(forecast_variants, dict):
            for vname, vcfg in forecast_variants.items():
                if vname == "teacher_perfect" or not hasattr(vcfg, "sigma_rel") or vcfg.sigma_rel <= 0.0:
                    continue
                noisy_variants[vname] = vcfg
                bills[f"teacher_{vname}"] = {"bill_total": 0.0, "energy_cost": 0.0, "demand_cost": 0.0, "peak_kw": 0.0}

        def _acc_bill(model_key: str, b: eil.BillBreakdown) -> None:
            bills[model_key]["bill_total"] += float(b.bill_total)
            bills[model_key]["energy_cost"] += float(b.energy_cost)
            bills[model_key]["demand_cost"] += float(b.demand_cost)
            bills[model_key]["peak_kw"] = max(float(bills[model_key]["peak_kw"]), float(b.peak_kw))

        roll_lists: dict[str, list[np.ndarray]] = {
            "load_kw": [],
            "is_peak_window": [],
            "energy_price": [],
            "grid_no_battery": [],
            "grid_bc_baseline": [],
            "grid_bc_tariff_aware": [],
            "grid_teacher": [],
            "P_batt_no_battery": [],
            "soc_no_battery": [],
            "P_batt_bc_baseline": [],
            "soc_bc_baseline": [],
            "P_batt_bc_tariff_aware": [],
            "soc_bc_tariff_aware": [],
            "P_batt_teacher": [],
            "soc_teacher": [],
        }
        if model_base_dagger is not None:
            roll_lists["grid_dagger_baseline"] = []
            roll_lists["P_batt_dagger_baseline"] = []
            roll_lists["soc_dagger_baseline"] = []
        if model_ta_dagger is not None:
            roll_lists["grid_dagger_tariff_aware"] = []
            roll_lists["P_batt_dagger_tariff_aware"] = []
            roll_lists["soc_dagger_tariff_aware"] = []
        roll_lists["grid_rule_tariff_aware"] = []
        roll_lists["P_batt_rule_tariff_aware"] = []
        roll_lists["soc_rule_tariff_aware"] = []
        if model_ta_mse_selected is not None:
            roll_lists["grid_bc_tariff_aware_mse_selected"] = []
            roll_lists["P_batt_bc_tariff_aware_mse_selected"] = []
            roll_lists["soc_bc_tariff_aware_mse_selected"] = []

        dfs: list[pd.DataFrame] = []
        teacher_solves_all: list[eil.MPCResult] = []
        monthly_rows: list[dict[str, Any]] = []

        def _add_month_row(*, month: int, model_key: str, b: eil.BillBreakdown) -> None:
            _fv_m = "perfect"
            if model_key.startswith("teacher_noisy_") or model_key.startswith("teacher_") and model_key != "teacher":
                _fv_m = model_key.replace("teacher_", "", 1)
            monthly_rows.append(
                {
                    "family": family_label,
                    "model": model_key,
                    "tariff": tname,
                    "split": "test",
                    "month": int(month),
                    "forecast_variant": _fv_m,
                    "bill_total": float(b.bill_total),
                    "energy_cost": float(b.energy_cost),
                    "demand_cost": float(b.demand_cost),
                    "peak_kw": float(b.peak_kw),
                }
            )

        for month, sl in eval_month_slices:
            ts_seg = ts_utc[sl]
            load_seg = load_kw[sl]
            tf_seg = tf[sl]
            c = _slice_compiled(c_full, sl)

            bill_nb = eil.compute_monthly_bill(
                grid_kw=load_seg,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                dt_hours=float(loaded.dt_hours),
            )
            _assert_bill_consistency(bill_nb)
            _assert_peak_consistency(bill=bill_nb, grid_kw=load_seg, is_peak_window=c.is_peak_window)
            _acc_bill("no_battery", bill_nb)
            _add_month_row(month=month, model_key="no_battery", b=bill_nb)

            rr_base = eil.rollout_policy(
                model=model_base,
                load_kw=load_seg,
                compiled=c,
                time_features=tf_seg,
                battery=battery,
                allow_grid_export=allow_export,
                initial_E_kwh=init_E,
                initial_max_peak_kw=0.0,
                feature_cfg=feature_cfg_base,
                scales=scales,
            )
            bill_base = eil.compute_monthly_bill(
                grid_kw=rr_base.grid_kw_clamped,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                dt_hours=float(loaded.dt_hours),
            )
            _assert_bill_consistency(bill_base)
            _assert_peak_consistency(bill=bill_base, grid_kw=rr_base.grid_kw_clamped, is_peak_window=c.is_peak_window)
            _acc_bill("bc_baseline", bill_base)
            _add_month_row(month=month, model_key="bc_baseline", b=bill_base)

            rr_ta = eil.rollout_policy(
                model=model_ta,
                load_kw=load_seg,
                compiled=c,
                time_features=tf_seg,
                battery=battery,
                allow_grid_export=allow_export,
                initial_E_kwh=init_E,
                initial_max_peak_kw=0.0,
                feature_cfg=feature_cfg_ta,
                scales=scales,
            )
            bill_ta = eil.compute_monthly_bill(
                grid_kw=rr_ta.grid_kw_clamped,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                dt_hours=float(loaded.dt_hours),
            )
            _assert_bill_consistency(bill_ta)
            _assert_peak_consistency(bill=bill_ta, grid_kw=rr_ta.grid_kw_clamped, is_peak_window=c.is_peak_window)
            _acc_bill("bc_tariff_aware", bill_ta)
            _add_month_row(month=month, model_key="bc_tariff_aware", b=bill_ta)

            rr_ta_mse = None
            if model_ta_mse_selected is not None:
                rr_ta_mse = eil.rollout_policy(
                    model=model_ta_mse_selected,
                    load_kw=load_seg,
                    compiled=c,
                    time_features=tf_seg,
                    battery=battery,
                    allow_grid_export=allow_export,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    feature_cfg=feature_cfg_ta,
                    scales=scales,
                )
                bill_ta_mse = eil.compute_monthly_bill(
                    grid_kw=rr_ta_mse.grid_kw_clamped,
                    energy_price_per_kwh=c.energy_price,
                    is_peak_window=c.is_peak_window,
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    dt_hours=float(loaded.dt_hours),
                )
                _assert_bill_consistency(bill_ta_mse)
                _assert_peak_consistency(bill=bill_ta_mse, grid_kw=rr_ta_mse.grid_kw_clamped, is_peak_window=c.is_peak_window)
                _acc_bill("bc_tariff_aware_mse_selected", bill_ta_mse)
                _add_month_row(month=month, model_key="bc_tariff_aware_mse_selected", b=bill_ta_mse)

            rr_rule = eil.rollout_rule_policy(
                load_kw=load_seg,
                compiled=c,
                battery=battery,
                allow_grid_export=allow_export,
                initial_E_kwh=init_E,
                initial_max_peak_kw=0.0,
                cfg=rule_cfg,
            )
            bill_rule = eil.compute_monthly_bill(
                grid_kw=rr_rule.grid_kw_clamped,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                dt_hours=float(loaded.dt_hours),
            )
            _assert_bill_consistency(bill_rule)
            _assert_peak_consistency(bill=bill_rule, grid_kw=rr_rule.grid_kw_clamped, is_peak_window=c.is_peak_window)
            _acc_bill("rule_tariff_aware", bill_rule)
            _add_month_row(month=month, model_key="rule_tariff_aware", b=bill_rule)

            rr_bd = None
            rr_td = None
            if model_base_dagger is not None:
                rr_bd = eil.rollout_policy(
                    model=model_base_dagger,
                    load_kw=load_seg,
                    compiled=c,
                    time_features=tf_seg,
                    battery=battery,
                    allow_grid_export=allow_export,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    feature_cfg=feature_cfg_base,
                    scales=scales,
                )
                bill_bd = eil.compute_monthly_bill(
                    grid_kw=rr_bd.grid_kw_clamped,
                    energy_price_per_kwh=c.energy_price,
                    is_peak_window=c.is_peak_window,
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    dt_hours=float(loaded.dt_hours),
                )
                _assert_bill_consistency(bill_bd)
                _assert_peak_consistency(bill=bill_bd, grid_kw=rr_bd.grid_kw_clamped, is_peak_window=c.is_peak_window)
                _acc_bill("dagger_baseline", bill_bd)
                _add_month_row(month=month, model_key="dagger_baseline", b=bill_bd)

            if model_ta_dagger is not None:
                rr_td = eil.rollout_policy(
                    model=model_ta_dagger,
                    load_kw=load_seg,
                    compiled=c,
                    time_features=tf_seg,
                    battery=battery,
                    allow_grid_export=allow_export,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    feature_cfg=feature_cfg_ta,
                    scales=scales,
                )
                bill_td = eil.compute_monthly_bill(
                    grid_kw=rr_td.grid_kw_clamped,
                    energy_price_per_kwh=c.energy_price,
                    is_peak_window=c.is_peak_window,
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    dt_hours=float(loaded.dt_hours),
                )
                _assert_bill_consistency(bill_td)
                _assert_peak_consistency(bill=bill_td, grid_kw=rr_td.grid_kw_clamped, is_peak_window=c.is_peak_window)
                _acc_bill("dagger_tariff_aware", bill_td)
                _add_month_row(month=month, model_key="dagger_tariff_aware", b=bill_td)

            teacher_rollout, teacher_solves = eil.run_teacher_receding_horizon(
                load_kw=load_seg,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                battery=battery,
                allow_grid_export=allow_export,
                teacher=teacher,
                horizon_steps=horizon_steps,
                initial_E_kwh=init_E,
                initial_max_peak_kw=0.0,
            )
            teacher_solves_all.extend(teacher_solves)
            bill_teacher = eil.compute_monthly_bill(
                grid_kw=teacher_rollout.grid_kw_clamped,
                energy_price_per_kwh=c.energy_price,
                is_peak_window=c.is_peak_window,
                demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                dt_hours=float(loaded.dt_hours),
            )
            _assert_bill_consistency(bill_teacher)
            _assert_peak_consistency(bill=bill_teacher, grid_kw=teacher_rollout.grid_kw_clamped, is_peak_window=c.is_peak_window)
            _acc_bill("teacher", bill_teacher)
            _add_month_row(month=month, model_key="teacher", b=bill_teacher)

            # Noisy teacher forecast variants
            for vname, vcfg in noisy_variants.items():
                _noise_seed = vcfg.seed * 10_000 + hash(name) % 10_000 + int(month)
                noisy_load = eil.apply_ar1_forecast_noise(
                    load_seg, sigma_rel=vcfg.sigma_rel, rho=vcfg.rho, seed=_noise_seed,
                )
                noisy_rollout, _ = eil.run_teacher_receding_horizon(
                    load_kw=load_seg,
                    energy_price_per_kwh=c.energy_price,
                    is_peak_window=c.is_peak_window,
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    battery=battery,
                    allow_grid_export=allow_export,
                    teacher=teacher,
                    horizon_steps=horizon_steps,
                    initial_E_kwh=init_E,
                    initial_max_peak_kw=0.0,
                    forecast_load_kw=noisy_load,
                )
                bill_noisy = eil.compute_monthly_bill(
                    grid_kw=noisy_rollout.grid_kw_clamped,
                    energy_price_per_kwh=c.energy_price,
                    is_peak_window=c.is_peak_window,
                    demand_charge_rate_kw=float(tariff.demand_charge_rate_kw),
                    dt_hours=float(loaded.dt_hours),
                )
                _acc_bill(f"teacher_{vname}", bill_noisy)
                _add_month_row(month=month, model_key=f"teacher_{vname}", b=bill_noisy)

            roll_lists["load_kw"].append(np.asarray(load_seg, dtype=float))
            roll_lists["is_peak_window"].append(np.asarray(c.is_peak_window, dtype=int))
            roll_lists["energy_price"].append(np.asarray(c.energy_price, dtype=float))
            roll_lists["grid_no_battery"].append(np.asarray(load_seg, dtype=float))
            roll_lists["grid_bc_baseline"].append(np.asarray(rr_base.grid_kw_clamped, dtype=float))
            roll_lists["grid_bc_tariff_aware"].append(np.asarray(rr_ta.grid_kw_clamped, dtype=float))
            roll_lists["grid_teacher"].append(np.asarray(teacher_rollout.grid_kw_clamped, dtype=float))
            roll_lists["P_batt_no_battery"].append(np.zeros_like(load_seg, dtype=float))
            roll_lists["soc_no_battery"].append(np.full_like(load_seg, float(init_E), dtype=float))
            roll_lists["P_batt_bc_baseline"].append(np.asarray(rr_base.P_batt_kw_safe, dtype=float))
            roll_lists["soc_bc_baseline"].append(np.asarray(rr_base.E_kwh[:-1], dtype=float))
            roll_lists["P_batt_bc_tariff_aware"].append(np.asarray(rr_ta.P_batt_kw_safe, dtype=float))
            roll_lists["soc_bc_tariff_aware"].append(np.asarray(rr_ta.E_kwh[:-1], dtype=float))
            roll_lists["grid_rule_tariff_aware"].append(np.asarray(rr_rule.grid_kw_clamped, dtype=float))
            roll_lists["P_batt_rule_tariff_aware"].append(np.asarray(rr_rule.P_batt_kw_safe, dtype=float))
            roll_lists["soc_rule_tariff_aware"].append(np.asarray(rr_rule.E_kwh[:-1], dtype=float))
            roll_lists["P_batt_teacher"].append(np.asarray(teacher_rollout.P_batt_kw, dtype=float))
            roll_lists["soc_teacher"].append(np.asarray(teacher_rollout.E_kwh[:-1], dtype=float))
            if model_base_dagger is not None and rr_bd is not None:
                roll_lists["grid_dagger_baseline"].append(np.asarray(rr_bd.grid_kw_clamped, dtype=float))
                roll_lists["P_batt_dagger_baseline"].append(np.asarray(rr_bd.P_batt_kw_safe, dtype=float))
                roll_lists["soc_dagger_baseline"].append(np.asarray(rr_bd.E_kwh[:-1], dtype=float))
            if model_ta_dagger is not None and rr_td is not None:
                roll_lists["grid_dagger_tariff_aware"].append(np.asarray(rr_td.grid_kw_clamped, dtype=float))
                roll_lists["P_batt_dagger_tariff_aware"].append(np.asarray(rr_td.P_batt_kw_safe, dtype=float))
                roll_lists["soc_dagger_tariff_aware"].append(np.asarray(rr_td.E_kwh[:-1], dtype=float))
            if model_ta_mse_selected is not None and rr_ta_mse is not None:
                roll_lists["grid_bc_tariff_aware_mse_selected"].append(np.asarray(rr_ta_mse.grid_kw_clamped, dtype=float))
                roll_lists["P_batt_bc_tariff_aware_mse_selected"].append(np.asarray(rr_ta_mse.P_batt_kw_safe, dtype=float))
                roll_lists["soc_bc_tariff_aware_mse_selected"].append(np.asarray(rr_ta_mse.E_kwh[:-1], dtype=float))

            if bool(il_cfg.get("write_eval_timeseries_csv", True)):
                dt_hours = float(loaded.dt_hours)
                is_peak = np.asarray(c.is_peak_window, dtype=int)
                price = np.asarray(c.energy_price, dtype=float)
                df = pd.DataFrame(
                    {
                        "timestamp": ts_seg.astype(str),
                        "t": np.arange(int(sl.start), int(sl.stop), dtype=int),
                        "family": family_label,
                        "tariff": str(tname),
                        "dt_hours": float(dt_hours),
                        "load_kw": np.asarray(load_seg, dtype=float),
                        "is_peak_window": is_peak,
                        "energy_price": price,
                        "minutes_to_window_start": np.asarray(c.minutes_to_window_start, dtype=int),
                        "minutes_to_window_end": np.asarray(c.minutes_to_window_end, dtype=int),
                        "demand_charge_rate_kw": float(tariff.demand_charge_rate_kw),
                    }
                )

                def _add_model(prefix: str, *, grid: np.ndarray, rr: Any | None, init_soc_kwh: float) -> None:
                    g = np.asarray(grid, dtype=float)
                    df[f"grid_kw_{prefix}"] = g
                    df[f"grid_kw_peak_window_{prefix}"] = g * is_peak.astype(float)
                    df[f"energy_cost_step_{prefix}"] = g * float(dt_hours) * price
                    df[f"energy_cost_cum_{prefix}"] = np.cumsum(df[f"energy_cost_step_{prefix}"].to_numpy(dtype=float))

                    _emax = battery.E_max_kwh
                    if rr is None:
                        df[f"P_batt_kw_raw_{prefix}"] = 0.0
                        df[f"P_batt_kw_safe_{prefix}"] = 0.0
                        df[f"soc_kwh_{prefix}"] = float(init_soc_kwh)
                        df[f"soc_pct_{prefix}"] = float(init_soc_kwh) / _emax * 100.0
                        df[f"soc_next_kwh_{prefix}"] = float(init_soc_kwh)
                        df[f"soc_next_pct_{prefix}"] = float(init_soc_kwh) / _emax * 100.0
                        peak_state = _compute_running_peak_kw(grid_kw=g, is_peak_window=is_peak)
                        df[f"current_max_peak_kw_{prefix}"] = peak_state[:-1]
                        df[f"current_max_peak_next_kw_{prefix}"] = peak_state[1:]
                    else:
                        _soc = np.asarray(rr.E_kwh[:-1], dtype=float)
                        _soc_next = np.asarray(rr.E_kwh[1:], dtype=float)
                        df[f"P_batt_kw_raw_{prefix}"] = np.asarray(rr.P_batt_kw_raw, dtype=float)
                        df[f"P_batt_kw_safe_{prefix}"] = np.asarray(rr.P_batt_kw_safe, dtype=float)
                        df[f"soc_kwh_{prefix}"] = _soc
                        df[f"soc_pct_{prefix}"] = _soc / _emax * 100.0
                        df[f"soc_next_kwh_{prefix}"] = _soc_next
                        df[f"soc_next_pct_{prefix}"] = _soc_next / _emax * 100.0
                        df[f"current_max_peak_kw_{prefix}"] = np.asarray(rr.current_max_peak_kw[:-1], dtype=float)
                        df[f"current_max_peak_next_kw_{prefix}"] = np.asarray(rr.current_max_peak_kw[1:], dtype=float)

                    dpeak = (
                        df[f"current_max_peak_next_kw_{prefix}"].to_numpy(dtype=float)
                        - df[f"current_max_peak_kw_{prefix}"].to_numpy(dtype=float)
                    )
                    dpeak = np.maximum(dpeak, 0.0)
                    df[f"demand_charge_step_{prefix}"] = float(tariff.demand_charge_rate_kw) * dpeak
                    df[f"demand_charge_cum_{prefix}"] = np.cumsum(df[f"demand_charge_step_{prefix}"].to_numpy(dtype=float))
                    df[f"bill_total_cum_{prefix}"] = (
                        df[f"energy_cost_cum_{prefix}"].to_numpy(dtype=float)
                        + df[f"demand_charge_cum_{prefix}"].to_numpy(dtype=float)
                    )

                _add_model("no_battery", grid=load_seg, rr=None, init_soc_kwh=float(init_E))
                _add_model("bc_baseline", grid=rr_base.grid_kw_clamped, rr=rr_base, init_soc_kwh=float(init_E))
                _add_model("bc_tariff_aware", grid=rr_ta.grid_kw_clamped, rr=rr_ta, init_soc_kwh=float(init_E))
                _add_model("rule_tariff_aware", grid=rr_rule.grid_kw_clamped, rr=rr_rule, init_soc_kwh=float(init_E))
                if model_base_dagger is not None and rr_bd is not None:
                    _add_model("dagger_baseline", grid=rr_bd.grid_kw_clamped, rr=rr_bd, init_soc_kwh=float(init_E))
                if model_ta_dagger is not None and rr_td is not None:
                    _add_model("dagger_tariff_aware", grid=rr_td.grid_kw_clamped, rr=rr_td, init_soc_kwh=float(init_E))
                if model_ta_mse_selected is not None and rr_ta_mse is not None:
                    _add_model("bc_tariff_aware_mse_selected", grid=rr_ta_mse.grid_kw_clamped, rr=rr_ta_mse, init_soc_kwh=float(init_E))

                gT = np.asarray(teacher_rollout.grid_kw_clamped, dtype=float)
                df["grid_kw_teacher"] = gT
                df["grid_kw_peak_window_teacher"] = gT * is_peak.astype(float)
                df["energy_cost_step_teacher"] = gT * float(dt_hours) * price
                df["energy_cost_cum_teacher"] = np.cumsum(df["energy_cost_step_teacher"].to_numpy(dtype=float))
                df["P_batt_kw_teacher"] = np.asarray(teacher_rollout.P_batt_kw, dtype=float)
                _soc_t = np.asarray(teacher_rollout.E_kwh[:-1], dtype=float)
                _soc_t_next = np.asarray(teacher_rollout.E_kwh[1:], dtype=float)
                df["soc_kwh_teacher"] = _soc_t
                df["soc_pct_teacher"] = _soc_t / battery.E_max_kwh * 100.0
                df["soc_next_kwh_teacher"] = _soc_t_next
                df["soc_next_pct_teacher"] = _soc_t_next / battery.E_max_kwh * 100.0
                df["current_max_peak_kw_teacher"] = np.asarray(teacher_rollout.current_max_peak_kw[:-1], dtype=float)
                df["current_max_peak_next_kw_teacher"] = np.asarray(teacher_rollout.current_max_peak_kw[1:], dtype=float)

                dpeakT = (
                    df["current_max_peak_next_kw_teacher"].to_numpy(dtype=float)
                    - df["current_max_peak_kw_teacher"].to_numpy(dtype=float)
                )
                dpeakT = np.maximum(dpeakT, 0.0)
                df["demand_charge_step_teacher"] = float(tariff.demand_charge_rate_kw) * dpeakT
                df["demand_charge_cum_teacher"] = np.cumsum(df["demand_charge_step_teacher"].to_numpy(dtype=float))
                df["bill_total_cum_teacher"] = (
                    df["energy_cost_cum_teacher"].to_numpy(dtype=float)
                    + df["demand_charge_cum_teacher"].to_numpy(dtype=float)
                )

                dfs.append(df)

        for model_name, b in bills.items():
            # Derive structured forecast_variant column from model name
            _fv = "perfect"
            if model_name.startswith("teacher_noisy_") or model_name.startswith("teacher_") and model_name != "teacher":
                _fv = model_name.replace("teacher_", "", 1)
            rows.append(
                {
                    "family": family_label,
                    "model": model_name,
                    "tariff": tname,
                    "split": "test",
                    "forecast_variant": _fv,
                    "bill_total": float(b["bill_total"]),
                    "energy_cost": float(b["energy_cost"]),
                    "demand_cost": float(b["demand_cost"]),
                    "peak_kw": float(b["peak_kw"]),
                }
            )

        monthly_rows_all.extend(monthly_rows)

        ts_solve = np.asarray([s.solver_log.solve_time_sec for s in teacher_solves_all], dtype=float)
        runtime_rows.append(
            {
                "family": family_label,
                "model": "teacher",
                "tariff": tname,
                "split": "test",
                "teacher_solve_ms_mean": float(np.mean(ts_solve) * 1e3) if ts_solve.size else 0.0,
                "teacher_solve_ms_p95": float(np.percentile(ts_solve, 95) * 1e3) if ts_solve.size else 0.0,
                "teacher_solver": str(teacher_solves_all[0].solver_log.solver_used) if teacher_solves_all else "",
                "teacher_used_fallback": bool(teacher_solves_all[0].solver_log.used_fallback_solver) if teacher_solves_all else False,
            }
        )

        roll = {
            "tariff_name": np.asarray([tname], dtype=object),
            "family": np.asarray([family_label], dtype=object),
            "dt_hours": np.asarray([float(loaded.dt_hours)], dtype=float),
        }
        for k, v_list in roll_lists.items():
            roll[k] = np.concatenate(v_list, axis=0) if v_list else np.zeros((0,), dtype=float)
        np.savez_compressed(metrics_dir / f"eval_rollouts_{family_label}_{tname}.npz", **roll)

        if bool(il_cfg.get("write_eval_timeseries_csv", True)):
            out_csv = metrics_dir / f"eval_timeseries_{family_label}_{tname}.csv"
            if not out_csv.exists():
                df_all = pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
                df_all.to_csv(out_csv, index=False)
                log.info("Wrote %s", str(out_csv))

        log.info("IL eval done | family=%s | tariff=%s", family_label, tname)

    tracker.end_step("il_eval" + family_suffix, ok=True, rows=int(len(rows)))

    return rows, runtime_rows, monthly_rows_all


def _run_il_pipeline(*, cfg: Mapping[str, Any], run_dir: Path, tracker: PipelineTracker) -> None:
    log = logging.getLogger("tail")

    il_cfg = cfg.get("il", {})
    if not isinstance(il_cfg, dict):
        il_cfg = {}

    data_root = Path(il_cfg.get("data_root", cfg.get("data_root", "data/processed/timeseries")))
    group = str(il_cfg.get("timeseries_group", cfg.get("timeseries_group", "")))
    if not group:
        raise ValueError("IL mode requires il.timeseries_group")

    ts_root = data_root / group
    if not ts_root.exists():
        raise ValueError(f"timeseries_group path not found: {ts_root}")

    building_id = il_cfg.get("building_id")
    if building_id is None:
        candidates = sorted(ts_root.glob("*.csv"))
        if not candidates:
            raise ValueError(f"no CSVs found under {ts_root}")
        building_id = candidates[0].stem

    path = ts_root / f"{building_id}.csv"

    tracker.start_step("il_load_timeseries", group=str(group), building_id=str(building_id))
    loaded = eil.load_processed_timeseries_csv(path)
    tracker.end_step("il_load_timeseries", ok=True, n=int(len(loaded.load_kw)), dt_hours=float(loaded.dt_hours))

    max_steps = il_cfg.get("max_steps")
    T_use = int(max_steps) if max_steps is not None else int(len(loaded.load_kw))
    if T_use <= 10:
        raise ValueError("il.max_steps too small")

    ts_utc = loaded.timestamps_utc[:T_use]
    load_kw = np.asarray(loaded.load_kw[:T_use], dtype=float)

    suite_tz = str(il_cfg.get("timezone", "UTC"))
    weekmask = str(il_cfg.get("weekmask", "MON_FRI"))
    suite = _resolve_tariff_suite(il_cfg=il_cfg)

    split_mode = str(il_cfg.get("split_mode", "threshold")).lower()
    if split_mode in {"seasonal", "seasonal_stratified", "paper"}:
        split = eil.split_by_month_lists(
            ts_utc,
            timezone=suite_tz,
            train_months=list(il_cfg.get("train_months", [1, 4, 6, 7, 10, 12])),
            val_months=list(il_cfg.get("val_months", [3, 9])),
            test_months=list(il_cfg.get("test_months", [2, 5, 8, 11])),
        )
    else:
        split = eil.split_by_month(
            ts_utc,
            timezone=suite_tz,
            val_start_month=int(il_cfg.get("val_start_month", 10)),
            test_start_month=int(il_cfg.get("test_start_month", 12)),
        )

    tf = eil.basic_time_features(ts_utc, timezone=suite_tz)

    month_slices = _month_slices(ts_utc, timezone=suite_tz)
    train_months = _months_from_idx(ts_utc, timezone=suite_tz, idx=split.train_idx)
    val_months = _months_from_idx(ts_utc, timezone=suite_tz, idx=split.val_idx)
    test_months = _months_from_idx(ts_utc, timezone=suite_tz, idx=split.test_idx)

    batt_cfg = il_cfg.get("battery", {})
    if not isinstance(batt_cfg, dict):
        batt_cfg = {}

    auto_size = bool(batt_cfg.get("auto_size", False))
    window_hours = float(batt_cfg.get("sizing_window_hours", 4.0))
    freeze_size = bool(batt_cfg.get("freeze_derived_size_per_building", False))
    reuse_frozen = bool(batt_cfg.get("reuse_frozen_size_across_suite", False))

    # Try to reuse a previously frozen battery sizing from a prior run.
    _frozen_path = run_dir / "battery" / "battery_sizing.json"
    _reused_frozen = False
    if reuse_frozen and _frozen_path.exists():
        try:
            import json as _json
            with open(_frozen_path, "r") as _fp:
                _prev = _json.load(_fp)
            E_max_kwh = float(_prev["E_max_kwh"])
            P_max_kw = float(_prev["P_max_kw"])
            _reused_frozen = True
            log.info("Battery sizing | reused frozen sizing from %s", str(_frozen_path))
        except Exception:
            _reused_frozen = False

    if not _reused_frozen:
        if auto_size:
            max_kwh = 0.0
            for month, sl in month_slices:
                if train_months and (int(month) not in train_months):
                    continue
                max_kwh = max(
                    max_kwh,
                    _max_energy_window_kwh(
                        load_kw[sl],
                        dt_hours=float(loaded.dt_hours),
                        window_hours=window_hours,
                    ),
                )
            E_max_kwh = max(max_kwh, 1e-6)
            P_max_kw = float(E_max_kwh)  # 1C sizing
        else:
            E_max_kwh = float(batt_cfg.get("E_max_kwh", 300.0))
            P_max_kw = float(batt_cfg.get("P_max_kw", 150.0))

    battery = eil.BatteryParams(
        E_max_kwh=float(E_max_kwh),
        P_max_kw=float(P_max_kw),
        eta_charge=float(batt_cfg.get("eta_charge", 0.95)),
        eta_discharge=float(batt_cfg.get("eta_discharge", 0.95)),
        dt_hours=float(loaded.dt_hours),
    )
    battery.validate()
    allow_export = bool(il_cfg.get("allow_grid_export", False))
    auto_init_soc = bool(batt_cfg.get("auto_initial_soc", False))
    init_E = float(0.5 * battery.E_max_kwh) if auto_init_soc else float(
        il_cfg.get("initial_E_kwh", 0.5 * battery.E_max_kwh)
    )

    log.info(
        "Battery sizing | auto_size=%s | freeze=%s | reused_frozen=%s | E_max_kwh=%.3f | P_max_kw=%.3f | window_hours=%.2f | init_E_kwh=%.3f",
        str(auto_size),
        str(freeze_size),
        str(_reused_frozen),
        float(battery.E_max_kwh),
        float(battery.P_max_kw),
        float(window_hours),
        float(init_E),
    )

    # Write battery sizing artifact
    battery_dir = run_dir / "battery"
    battery_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        battery_dir / "battery_sizing.json",
        {
            "auto_size": bool(auto_size),
            "sizing_window_hours": float(window_hours),
            "E_max_kwh": float(battery.E_max_kwh),
            "P_max_kw": float(battery.P_max_kw),
            "eta_charge": float(battery.eta_charge),
            "eta_discharge": float(battery.eta_discharge),
            "dt_hours": float(battery.dt_hours),
            "initial_E_kwh": float(init_E),
            "auto_initial_soc": bool(auto_init_soc),
        },
    )

    # Persist derived sizing in config snapshot for auditability.
    try:
        if isinstance(cfg, dict):
            cfg_il = cfg.get("il") if isinstance(cfg.get("il"), dict) else None
            if cfg_il is not None:
                cfg_batt = cfg_il.get("battery") if isinstance(cfg_il.get("battery"), dict) else None
                if cfg_batt is not None:
                    cfg_batt["derived_E_max_kwh"] = float(battery.E_max_kwh)
                    cfg_batt["derived_P_max_kw"] = float(battery.P_max_kw)
                    cfg_batt["derived_init_E_kwh"] = float(init_E)
                    cfg_batt["derived_sizing_window_hours"] = float(window_hours)
                    write_json(run_dir / "config_snapshot.json", cfg)
    except Exception:
        pass

    teacher_cfg = il_cfg.get("teacher", {})
    if not isinstance(teacher_cfg, dict):
        teacher_cfg = {}
    teacher = eil.TeacherConfig(
        solver=str(teacher_cfg.get("solver", "GUROBI")),
        allow_solver_fallback=bool(teacher_cfg.get("allow_solver_fallback", True)),
        fallback_solver_order=tuple(teacher_cfg.get("fallback_solver_order", ("OSQP", "ECOS", "SCS"))),
        time_limit_sec=(float(teacher_cfg["time_limit_sec"]) if teacher_cfg.get("time_limit_sec") is not None else None),
        lambda_batt_power=float(teacher_cfg.get("lambda_batt_power", 1e-3)),
        gurobi_threads=int(teacher_cfg.get("gurobi_threads", 1)),
    )
    horizon_steps = int(teacher_cfg.get("horizon_steps", 96))

    compiled: dict[str, tuple[eil.TariffIR, eil.CompiledTariff]] = {}
    for name, tariff in suite.items():
        compiled[name] = (tariff, eil.compile_tariff(tariff, ts_utc))

    feature_cfg_base = _resolve_feature_config(il_cfg, default_include_tariff=False)
    feature_cfg_ta = _resolve_feature_config(il_cfg, default_include_tariff=True)
    forecast_variants = _resolve_forecast_variants(il_cfg, seed=int(cfg.get("seed", 1337)))

    train_idx = split.train_idx
    if train_idx.size == 0:
        raise ValueError("empty train split")

    load_scale = float(np.max(load_kw[train_idx]))
    load_scale = max(load_scale, 1.0)

    # Resolve family plans: drives train/eval tariff selection per family
    family_plans = _resolve_family_plans(il_cfg=il_cfg, suite=suite)
    if not family_plans:
        family_plans = [{"family_id": None, "train_tariffs": ["A", "B"], "eval_tariffs": ["C", "C_flat"]}]

    # Stage switches: allow skipping teacher / train / eval phases
    _suite_cfg = cfg.get("suite", {})
    if not isinstance(_suite_cfg, dict):
        _suite_cfg = {}
    _stages_cfg = _suite_cfg.get("stages", {})
    if not isinstance(_stages_cfg, dict):
        _stages_cfg = {}
    _stage_teacher = bool(_stages_cfg.get("run_teacher", True))
    _stage_train = bool(_stages_cfg.get("run_train", True))
    _stage_eval = bool(_stages_cfg.get("run_eval", True))
    _stage_aggregate = bool(_stages_cfg.get("run_aggregate", True))
    _stage_figures = bool(_stages_cfg.get("run_figures", True))

    # --- Suite experiment dispatch ---
    _suite_experiments: list[dict[str, Any]] = []
    _raw_exps = _suite_cfg.get("experiments", [])
    if isinstance(_raw_exps, list):
        _suite_experiments = [e for e in _raw_exps if isinstance(e, dict) and e.get("enabled", True)]

    # If no experiments defined, run once with the default config (experiment=None).
    _experiment_iter: list[dict[str, Any] | None] = _suite_experiments if _suite_experiments else [None]  # type: ignore[list-item]

    for _exp in _experiment_iter:
        if _exp is not None:
            _exp_id = str(_exp.get("id", "unnamed"))
            _exp_run_dir = run_dir / "experiments" / _exp_id
            _exp_run_dir.mkdir(parents=True, exist_ok=True)

            _exp_il_cfg = copy.deepcopy(il_cfg)
            # Override feature set
            if _exp.get("feature_set"):
                if "features" not in _exp_il_cfg or not isinstance(_exp_il_cfg.get("features"), dict):
                    _exp_il_cfg["features"] = {}
                _exp_il_cfg["features"]["active_feature_set"] = str(_exp["feature_set"])
            # Override active tariff families
            if _exp.get("active_tariff_families"):
                if "experiment" not in _exp_il_cfg or not isinstance(_exp_il_cfg.get("experiment"), dict):
                    _exp_il_cfg["experiment"] = {}
                _exp_il_cfg["experiment"]["active_tariff_families"] = list(_exp["active_tariff_families"])
            # Override dagger
            if "dagger_enabled" in _exp:
                if "dagger" not in _exp_il_cfg or not isinstance(_exp_il_cfg.get("dagger"), dict):
                    _exp_il_cfg["dagger"] = {}
                _exp_il_cfg["dagger"]["enabled"] = bool(_exp["dagger_enabled"])

            log.info("Experiment dispatch | id=%s | feature_set=%s | families=%s | dagger=%s",
                     _exp_id, str(_exp.get("feature_set")), str(_exp.get("active_tariff_families")),
                     str(_exp.get("dagger_enabled")))

            # Re-resolve per-experiment configs
            _exp_feature_base = _resolve_feature_config(_exp_il_cfg, default_include_tariff=False)
            _exp_feature_ta = _resolve_feature_config(_exp_il_cfg, default_include_tariff=True)
            _exp_forecast_variants = _resolve_forecast_variants(_exp_il_cfg, seed=int(cfg.get("seed", 1337)))
            # If experiment specifies a single forecast variant, filter
            _fv_name = str(_exp.get("forecast_variant", "")).strip()
            if _fv_name and _fv_name != "teacher_perfect":
                if _fv_name in _exp_forecast_variants:
                    _exp_forecast_variants = {_fv_name: _exp_forecast_variants[_fv_name]}
            elif _fv_name == "teacher_perfect":
                _exp_forecast_variants = {}  # no noisy variants

            _exp_family_plans = _resolve_family_plans(il_cfg=_exp_il_cfg, suite=suite)
            if not _exp_family_plans:
                _exp_family_plans = family_plans
        else:
            _exp_id = "default"
            _exp_run_dir = run_dir
            _exp_il_cfg = il_cfg
            _exp_feature_base = feature_cfg_base
            _exp_feature_ta = feature_cfg_ta
            _exp_forecast_variants = forecast_variants
            _exp_family_plans = family_plans

        # Accumulate metrics rows across all family plans (per experiment)
        all_rows: list[dict[str, Any]] = []
        all_runtime_rows: list[dict[str, Any]] = []
        all_monthly_rows: list[dict[str, Any]] = []

        for plan_idx, plan in enumerate(_exp_family_plans):
            plan_family_id = plan.get("family_id")
            train_tariffs = [str(x) for x in plan.get("train_tariffs", ["A", "B"])]
            eval_tariffs = [str(x) for x in plan.get("eval_tariffs", ["C", "C_flat"])]
            family_suffix = f"_{plan_family_id}" if plan_family_id else ""
            log.info("Family plan start | exp=%s | family=%s | train=%s | eval=%s", _exp_id, str(plan_family_id), str(train_tariffs), str(eval_tariffs))

            # Compute price scale from this family's train tariffs
            price_vals = []
            for tname in train_tariffs:
                if tname in compiled:
                    price_vals.append(float(compiled[tname][1].energy_price.max()))
            price_scale = max(max(price_vals) if price_vals else 0.10, 1e-6)
            scales = eil.FeatureScales(load_kw_scale=load_scale, price_scale=price_scale)

            rows, runtime_rows, monthly_rows = _run_family_plan(
                cfg=cfg, il_cfg=_exp_il_cfg, run_dir=_exp_run_dir, tracker=tracker,
                train_tariffs=train_tariffs, eval_tariffs=eval_tariffs,
                family_suffix=family_suffix, plan_family_id=plan_family_id,
                compiled=compiled, ts_utc=ts_utc, load_kw=load_kw, tf=tf,
                battery=battery, teacher=teacher, horizon_steps=horizon_steps,
                month_slices=month_slices, train_months=train_months,
                val_months=val_months, test_months=test_months,
                allow_export=allow_export, init_E=init_E, auto_size=auto_size,
                window_hours=window_hours, loaded=loaded, T_use=T_use,
                split=split, scales=scales, train_idx=train_idx,
                feature_cfg_base=_exp_feature_base, feature_cfg_ta=_exp_feature_ta,
                forecast_variants=_exp_forecast_variants, suite_tz=suite_tz,
                run_teacher=_stage_teacher, run_train=_stage_train, run_eval=_stage_eval,
            )
            all_rows.extend(rows)
            all_runtime_rows.extend(runtime_rows)
            all_monthly_rows.extend(monthly_rows)

        # Write accumulated metrics across all family plans
        metrics_dir = _exp_run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        dfm = pd.DataFrame(all_rows)
        if not dfm.empty and "bill_total" in dfm.columns:
            dfm["savings_vs_no_battery"] = 0.0
            dfm["optimality_gap_vs_teacher"] = 0.0

            group_cols = ["tariff"]
            if "family" in dfm.columns:
                group_cols = ["family", "tariff"]
            for _, dft in dfm.groupby(group_cols, sort=False):
                tariff_name = str(dft["tariff"].iloc[0])
                family_value = str(dft["family"].iloc[0]) if "family" in dft.columns else None
                try:
                    bill_nb = float(dft.loc[dft["model"] == "no_battery", "bill_total"].iloc[0])
                except Exception:
                    bill_nb = float("nan")
                try:
                    bill_teacher = float(dft.loc[dft["model"] == "teacher", "bill_total"].iloc[0])
                except Exception:
                    bill_teacher = float("nan")

                if bill_nb == bill_nb and bill_nb > 0:
                    mask = dfm["tariff"] == tariff_name
                    if family_value is not None:
                        mask = mask & (dfm["family"] == family_value)
                    dfm.loc[mask, "savings_vs_no_battery"] = (bill_nb - dfm.loc[mask, "bill_total"]) / bill_nb

                eps = 1e-6
                if bill_teacher == bill_teacher:
                    denom = max(float(bill_teacher), eps)
                    mask = dfm["tariff"] == tariff_name
                    if family_value is not None:
                        mask = mask & (dfm["family"] == family_value)
                    dfm.loc[mask, "optimality_gap_vs_teacher"] = (dfm.loc[mask, "bill_total"] - bill_teacher) / denom

        dfm.to_csv(metrics_dir / "metrics.csv", index=False)
        write_json(metrics_dir / "metrics.json", {"rows": dfm.to_dict(orient="records")})

        if all_monthly_rows:
            df_monthly = pd.DataFrame(all_monthly_rows)
            df_monthly.to_csv(metrics_dir / "metrics_monthly.csv", index=False)

            try:
                nb = df_monthly[df_monthly["model"] == "no_battery"][
                    ["family", "tariff", "month", "bill_total"]
                ].rename(columns={"bill_total": "bill_total_no_batt"})
                df_monthly = df_monthly.merge(nb, on=["family", "tariff", "month"], how="left")
                df_monthly["savings_vs_no_battery"] = (
                    df_monthly["bill_total_no_batt"] - df_monthly["bill_total"]
                ) / df_monthly["bill_total_no_batt"]
            except Exception:
                pass

            summary = (
                df_monthly.groupby(["family", "tariff", "model"], sort=False)
                .agg(
                    n_months=("month", "nunique"),
                    bill_total_mean=("bill_total", "mean"),
                    bill_total_std=("bill_total", "std"),
                    energy_cost_mean=("energy_cost", "mean"),
                    energy_cost_std=("energy_cost", "std"),
                    demand_cost_mean=("demand_cost", "mean"),
                    demand_cost_std=("demand_cost", "std"),
                    peak_kw_mean=("peak_kw", "mean"),
                    peak_kw_std=("peak_kw", "std"),
                )
                .reset_index()
            )
            if "savings_vs_no_battery" in df_monthly.columns:
                sv = (
                    df_monthly.groupby(["family", "tariff", "model"], sort=False)["savings_vs_no_battery"]
                    .mean()
                    .reset_index()
                    .rename(columns={"savings_vs_no_battery": "savings_vs_no_battery_mean"})
                )
                summary = summary.merge(sv, on=["family", "tariff", "model"], how="left")

            summary.to_csv(metrics_dir / "metrics_summary.csv", index=False)

        dfr = pd.DataFrame(all_runtime_rows)
        if not dfr.empty:
            dfr.to_csv(metrics_dir / "runtimes.csv", index=False)
            write_json(metrics_dir / "runtimes.json", {"rows": all_runtime_rows})

        # Figures
        if _stage_figures and bool(_exp_il_cfg.get("make_figures", True)):
            try:
                _force_figs = bool(cfg.get("_force_figures", False))
                _make_figures_from_metrics_artifacts(run_dir=_exp_run_dir, force=_force_figs)
            except Exception as e:
                log.info("Figure generation skipped (exp=%s): %s", _exp_id, str(e))



def _prepare_building_cfg(
    cfg: Mapping[str, Any],
    building_id: str,
    manifest_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_cfg = copy.deepcopy(dict(cfg))
    il_cfg = local_cfg.get("il")
    if not isinstance(il_cfg, dict):
        il_cfg = {}
    il_cfg = copy.deepcopy(il_cfg)
    il_cfg.pop("building_ids", None)
    il_cfg.pop("building_manifest", None)
    il_cfg["building_id"] = building_id
    if manifest_entry:
        if manifest_entry.get("archetype"):
            il_cfg["archetype"] = str(manifest_entry["archetype"])
        if manifest_entry.get("timeseries_group"):
            il_cfg["timeseries_group"] = str(manifest_entry["timeseries_group"])
    local_cfg["il"] = il_cfg
    return local_cfg


def _run_building_worker(
    cfg: Mapping[str, Any],
    run_dir: str,
    building_id: str,
    per_run_workers: int,
    log_queue: mp.Queue | None,
    manifest_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_run_dir = Path(run_dir)
    _configure_worker_logging(run_dir=local_run_dir, log_queue=log_queue, building_id=str(building_id))
    log = logging.getLogger("tail")

    try:
        if torch is not None:
            torch.set_num_threads(int(per_run_workers))
        os.environ.setdefault("OMP_NUM_THREADS", str(int(per_run_workers)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(per_run_workers)))
    except Exception:
        pass

    local_cfg = _prepare_building_cfg(cfg, str(building_id), manifest_entry=manifest_entry)
    write_json(local_run_dir / "config_snapshot.json", local_cfg)
    tracker = PipelineTracker(run_dir=local_run_dir, cfg=local_cfg, steps={})

    try:
        _run_il_pipeline(cfg=local_cfg, run_dir=local_run_dir, tracker=tracker)
        tracker.mark_completed()
        log.info("BUILDING completed | building_id=%s | run_dir=%s", str(building_id), str(local_run_dir))
        return {"building_id": str(building_id), "ok": True, "error": None}
    except Exception as e:  # noqa: BLE001
        tracker.mark_failed(str(e))
        log.exception("BUILDING failed | building_id=%s", str(building_id))
        return {"building_id": str(building_id), "ok": False, "error": str(e)}


def _detach_and_exit(args: argparse.Namespace, run_dir: Path) -> None:
    """Re-exec this script under nohup+setsid; parent prints PID and exits."""
    import signal
    import subprocess

    nohup_out = run_dir / "nohup.out"
    child_argv = [sys.executable, "-u", __file__, "--config", str(args.config), "--run-dir", str(run_dir)]
    if args.make_figures_only:
        child_argv.append("--make-figures-only")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(nohup_out, "w") as outf:
        proc = subprocess.Popen(
            child_argv,
            stdout=outf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    # Ignore HUP in the parent so it can print and exit cleanly
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    print(f"detached  pid={proc.pid}")
    print(f"run_dir   {run_dir}")
    print(f"tail -f   {run_dir / 'run.log'}")
    print(f"status    cat {run_dir / 'status.json'}")
    sys.exit(0)


def main() -> None:
    args = _parse_args()
    cfg = read_yaml(Path(args.config))

    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    results_dir = Path(cfg.get("results_dir", "results"))
    runs_root = results_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        candidate = Path(str(args.run_dir))
        if candidate.is_absolute():
            run_dir = candidate
        else:
            cand_str = str(candidate).replace("\\", "/")
            if cand_str.startswith("results/runs/") or cand_str == "results/runs":
                run_dir = Path(cand_str)
            else:
                run_dir = runs_root / candidate
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_prefix = utc_run_id()
        seq = 1
        while (runs_root / f"{run_prefix}_{seq:04d}").exists():
            seq += 1
        run_dir = runs_root / f"{run_prefix}_{seq:04d}"
        run_dir.mkdir(parents=True, exist_ok=False)

    if args.detach:
        _detach_and_exit(args, run_dir)

    _configure_run_logging(run_dir=run_dir)
    log = logging.getLogger("tail")
    log.info("RUN created | run_dir=%s | config=%s", str(run_dir), str(args.config))

    # Convenience link: results/runs/latest -> this run (best effort)
    try:
        latest = runs_root / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        rel_target = os.path.relpath(str(run_dir), str(runs_root))
        os.symlink(rel_target, latest)
    except Exception:
        pass

    write_json(run_dir / "config_snapshot.json", cfg)

    if args.force_figures:
        cfg["_force_figures"] = True

    tracker = PipelineTracker(run_dir=run_dir, cfg=cfg, steps={})

    pipeline_mode = str(cfg.get("pipeline", "il")).strip().lower()
    if pipeline_mode not in {"il", "ta_il", "ta-il", "imitation", "imitation_learning"}:
        raise ValueError("This consolidated repo only supports pipeline: il")

    try:
        if args.make_figures_only:
            tracker.start_step("make_figures")
            _make_figures_from_metrics_artifacts(run_dir=run_dir, force=args.force_figures)
            tracker.end_step("make_figures", ok=True)
        else:
            il_cfg = cfg.get("il", {})
            if not isinstance(il_cfg, dict):
                il_cfg = {}

            building_entries = _normalize_building_entries(il_cfg)
            if building_entries:
                # Read stage switches for multi-building aggregation/figures
                _mb_suite_cfg = cfg.get("suite", {})
                if not isinstance(_mb_suite_cfg, dict):
                    _mb_suite_cfg = {}
                _mb_stages_cfg = _mb_suite_cfg.get("stages", {})
                if not isinstance(_mb_stages_cfg, dict):
                    _mb_stages_cfg = {}
                _stage_aggregate = bool(_mb_stages_cfg.get("run_aggregate", True))

                ids = [e["building_id"] for e in building_entries]
                max_workers, per_run_workers, cpu_cores = _resolve_parallelism(
                    il_cfg=il_cfg,
                    n_buildings=len(ids),
                )

                # Build a lookup for manifest metadata
                manifest_lookup = {e["building_id"]: e for e in building_entries}

                tracker.start_step(
                    "multi_building",
                    n_buildings=int(len(ids)),
                    max_workers=int(max_workers),
                    per_run_workers=int(per_run_workers),
                    building_ids=ids,
                    completed=0,
                    failed=0,
                )

                log.info(
                    "MULTI-BUILDING start | n=%d | max_workers=%d | per_run_workers=%d | cpu_cores=%d",
                    int(len(ids)),
                    int(max_workers),
                    int(per_run_workers),
                    int(cpu_cores),
                )

                start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
                ctx = mp.get_context(start_method)
                manager = mp.Manager()
                log_queue: mp.Queue | None = manager.Queue()
                queue_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
                queue_handler.setLevel(logging.INFO)
                queue_handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S",
                    )
                )
                listener = logging.handlers.QueueListener(log_queue, queue_handler)
                listener.start()

                failures: list[dict[str, Any]] = []
                completed = 0

                try:
                    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                        futures = {}
                        skipped_buildings: list[str] = []
                        for bid in ids:
                            building_dir = run_dir / f"building_{bid}"
                            # Resume: skip buildings whose status.json shows completed
                            _bld_status_path = building_dir / "status.json"
                            if _bld_status_path.exists():
                                try:
                                    import json as _json_check
                                    with open(_bld_status_path, "r") as _sf:
                                        _bld_status = _json_check.load(_sf)
                                    if _bld_status.get("status") == "completed":
                                        completed += 1
                                        skipped_buildings.append(str(bid))
                                        log.info("MULTI-BUILDING CACHED | building=%s | already completed, skipping", str(bid))
                                        continue
                                except Exception:
                                    pass  # status file unreadable, re-run this building

                            futures[
                                executor.submit(
                                    _run_building_worker,
                                    cfg,
                                    str(building_dir),
                                    str(bid),
                                    int(per_run_workers),
                                    log_queue,
                                    manifest_lookup.get(bid),
                                )
                            ] = bid

                        if skipped_buildings:
                            log.info(
                                "MULTI-BUILDING resume | skipped %d already-completed buildings: %s",
                                len(skipped_buildings),
                                ", ".join(skipped_buildings[:10]),
                            )

                        for fut in cf.as_completed(futures):
                            bid = futures[fut]
                            try:
                                result = fut.result()
                            except Exception as e:  # noqa: BLE001
                                result = {"building_id": str(bid), "ok": False, "error": str(e)}
                            completed += 1
                            if not result.get("ok", False):
                                failures.append(result)
                            tracker.update_step(
                                "multi_building",
                                completed=int(completed),
                                failed=int(len(failures)),
                                last_building=str(bid),
                                last_status=("failed" if not result.get("ok", False) else "completed"),
                            )
                            log.info(
                                "MULTI-BUILDING progress | completed=%d/%d | failed=%d",
                                int(completed),
                                int(len(ids)),
                                int(len(failures)),
                            )
                finally:
                    listener.stop()
                    manager.shutdown()

                tracker.end_step(
                    "multi_building",
                    ok=(len(failures) == 0),
                    completed=int(completed),
                    failed=int(len(failures)),
                    errors=[f.get("error") for f in failures[:3]],
                )

                if failures:
                    first_error = failures[0].get("error", "unknown error")
                    tracker.mark_failed(str(first_error))
                    log.error("MULTI-BUILDING failed | first_error=%s", str(first_error))
                    raise RuntimeError(str(first_error))

                log.info("MULTI-BUILDING completed | n=%d", int(len(ids)))

                # Aggregate per-building results at run root
                if _stage_aggregate:
                    _aggregate_multi_building_results(run_dir=run_dir)
                    log.info("MULTI-BUILDING aggregation done | run_dir=%s", str(run_dir))
                else:
                    log.info("MULTI-BUILDING aggregation skipped (run_aggregate=false)")
            else:
                _run_il_pipeline(cfg=cfg, run_dir=run_dir, tracker=tracker)
        tracker.mark_completed()
        log.info("RUN completed | run_dir=%s", str(run_dir))
    except Exception as e:  # noqa: BLE001
        tracker.mark_failed(str(e))
        log.exception("RUN failed")
        raise


if __name__ == "__main__":
    main()
