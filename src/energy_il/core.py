from __future__ import annotations

"""Consolidated TA-IL core module.

This file consolidates the previously modular implementation (tariffs/billing/env/teacher/data)
into a single import surface to keep the repository to <=5 Python files (excluding tests).

Public API is kept stable at the level used by `tail.py` and `tests/`.
"""

from dataclasses import dataclass
from datetime import time
from time import perf_counter
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]


# -----------------------------
# utils
# -----------------------------


def parse_hhmm(s: str) -> tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time string: {s!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time string: {s!r}")
    return hour, minute


def hhmm_to_minutes(s: str) -> int:
    hour, minute = parse_hhmm(s)
    return hour * 60 + minute


def ensure_zoneinfo(tz: str | ZoneInfo) -> ZoneInfo:
    if isinstance(tz, ZoneInfo):
        return tz
    return ZoneInfo(str(tz))


def minutes_until(target_minute_of_day: int, current_minute_of_day: int) -> int:
    target = int(target_minute_of_day)
    cur = int(current_minute_of_day)
    if not (0 <= target < 1440 and 0 <= cur < 1440):
        raise ValueError("minute_of_day must be in [0,1440)")
    if cur <= target:
        return target - cur
    return 1440 - cur + target


# -----------------------------
# data
# -----------------------------


@dataclass(frozen=True)
class LoadedTimeseries:
    timestamps_utc: pd.DatetimeIndex
    load_kw: np.ndarray
    dt_hours: float


def load_processed_timeseries_csv(path: str | Any) -> LoadedTimeseries:
    """Load a processed timeseries CSV.

    Supports both of these formats:
    - timestamp,load_kwh  (interval energy per dt)
    - timestamp,load_kw   (average power)

    Unit detection matches the tests and copilot.md: if `load_kw` exists, it is used.
    Otherwise `load_kwh` is converted using inferred dt.
    """

    path = str(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must have timestamp column")

    ts = pd.to_datetime(df["timestamp"], utc=True)
    if ts.isna().any():
        raise ValueError("timestamp contains NaT")
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be increasing")

    if len(ts) < 2:
        raise ValueError("timeseries too short")

    # Ensure numpy datetime64[ns] for timedelta math
    arr = ts.to_numpy(dtype="datetime64[ns]")
    deltas = (arr[1:] - arr[:-1]) / np.timedelta64(1, "s")
    if deltas.size == 0:
        raise ValueError("timeseries too short")
    dt_seconds = float(deltas[0])
    if dt_seconds <= 0:
        raise ValueError("invalid dt")
    if not np.allclose(deltas, dt_seconds, rtol=0.0, atol=1e-6):
        raise ValueError("Non-uniform timesteps detected")
    dt_hours = dt_seconds / 3600.0

    if "load_kw" in df.columns:
        load_kw = df["load_kw"].to_numpy(dtype=float)
        return LoadedTimeseries(timestamps_utc=pd.DatetimeIndex(ts), load_kw=load_kw, dt_hours=float(dt_hours))

    if "load_kwh" not in df.columns:
        raise ValueError("CSV must have load_kw or load_kwh")

    load_kwh = df["load_kwh"].to_numpy(dtype=float)
    # Convert interval energy to average power
    load_kw = load_kwh / float(dt_hours)
    return LoadedTimeseries(timestamps_utc=pd.DatetimeIndex(ts), load_kw=load_kw, dt_hours=float(dt_hours))


def basic_time_features(timestamps_utc: pd.DatetimeIndex, *, timezone: str) -> np.ndarray:
    tz = ensure_zoneinfo(timezone)
    if timestamps_utc.tz is None:
        raise ValueError("timestamps_utc must be timezone-aware")
    local = timestamps_utc.tz_convert(tz)

    month = local.month.to_numpy(dtype=int)
    minute_of_day = (local.hour * 60 + local.minute).to_numpy(dtype=int)
    day_of_week = local.dayofweek.to_numpy(dtype=int)
    is_holiday = np.zeros_like(day_of_week, dtype=int)

    days_in_month = local.days_in_month.to_numpy(dtype=int)
    minutes_since_month_start = (
        (local.day.to_numpy(dtype=int) - 1) * 1440 + minute_of_day
    ).astype(float)
    month_progress = minutes_since_month_start / np.maximum(days_in_month.astype(float) * 1440.0, 1.0)

    return np.stack([month, minute_of_day, day_of_week, is_holiday, month_progress], axis=1).astype(np.float32)


@dataclass(frozen=True)
class MonthSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def split_by_month(
    timestamps_utc: pd.DatetimeIndex,
    *,
    timezone: str,
    val_start_month: int,
    test_start_month: int,
) -> MonthSplit:
    if timestamps_utc.tz is None:
        raise ValueError("timestamps_utc must be timezone-aware (UTC)")

    if not (1 <= val_start_month <= 12 and 1 <= test_start_month <= 12):
        raise ValueError("val_start_month/test_start_month must be in 1..12")
    if test_start_month <= val_start_month:
        raise ValueError("test_start_month must be > val_start_month")

    tz = ensure_zoneinfo(timezone)
    ts_local = timestamps_utc.tz_convert(tz)
    months = ts_local.month.to_numpy(dtype=int)

    train = np.where(months < int(val_start_month))[0]
    val = np.where((months >= int(val_start_month)) & (months < int(test_start_month)))[0]
    test = np.where(months >= int(test_start_month))[0]

    return MonthSplit(train_idx=train, val_idx=val, test_idx=test)


def split_by_month_lists(
    timestamps_utc: pd.DatetimeIndex,
    *,
    timezone: str,
    train_months: list[int],
    val_months: list[int],
    test_months: list[int],
) -> MonthSplit:
    if timestamps_utc.tz is None:
        raise ValueError("timestamps_utc must be timezone-aware (UTC)")

    def _validate(ms: list[int], name: str) -> list[int]:
        out: list[int] = []
        for m in ms:
            mi = int(m)
            if not (1 <= mi <= 12):
                raise ValueError(f"{name} months must be in 1..12")
            out.append(mi)
        return list(dict.fromkeys(out))

    train_m = _validate(train_months, "train")
    val_m = _validate(val_months, "val")
    test_m = _validate(test_months, "test")

    overlap = set(train_m) & set(val_m) | set(train_m) & set(test_m) | set(val_m) & set(test_m)
    if overlap:
        raise ValueError(f"month sets must be disjoint; overlap={sorted(overlap)}")

    tz = ensure_zoneinfo(timezone)
    ts_local = timestamps_utc.tz_convert(tz)
    months = ts_local.month.to_numpy(dtype=int)

    train = np.where(np.isin(months, np.asarray(train_m, dtype=int)))[0]
    val = np.where(np.isin(months, np.asarray(val_m, dtype=int)))[0]
    test = np.where(np.isin(months, np.asarray(test_m, dtype=int)))[0]

    return MonthSplit(train_idx=train, val_idx=val, test_idx=test)


# -----------------------------
# tariffs
# -----------------------------


Weekmask = Literal["MON_FRI", "ALL_DAYS"]
DemandChargeMode = Literal["PEAK_WINDOW_MONTHLY_MAX"]


@dataclass(frozen=True)
class PeakWindow:
    start: str
    end: str

    @property
    def start_minute(self) -> int:
        return hhmm_to_minutes(self.start)

    @property
    def end_minute(self) -> int:
        return hhmm_to_minutes(self.end)

    def validate(self, *, allow_wraparound: bool = False) -> None:
        s = self.start_minute
        e = self.end_minute
        if s == e:
            raise ValueError("peak_window start and end must differ")
        if not allow_wraparound and e <= s:
            raise ValueError(
                f"wrap-around peak windows are not allowed here (start={self.start}, end={self.end})"
            )


@dataclass(frozen=True)
class CalendarMask:
    months: tuple[int, ...] = tuple(range(1, 13))
    weekmask: Weekmask = "MON_FRI"

    def validate(self) -> None:
        if not self.months:
            raise ValueError("calendar mask must include at least one month")
        for month in self.months:
            if not (1 <= int(month) <= 12):
                raise ValueError(f"invalid month in calendar mask: {month}")
        if self.weekmask not in ("MON_FRI", "ALL_DAYS"):
            raise ValueError(f"invalid weekmask: {self.weekmask}")


@dataclass(frozen=True)
class DemandWindowRule:
    start: str
    end: str

    @property
    def start_minute(self) -> int:
        return hhmm_to_minutes(self.start)

    @property
    def end_minute(self) -> int:
        return hhmm_to_minutes(self.end)

    def validate(self) -> None:
        PeakWindow(start=self.start, end=self.end).validate(allow_wraparound=False)


@dataclass(frozen=True)
class EnergyPriceRule:
    start: str
    end: str
    price: float
    label: str = ""

    @property
    def start_minute(self) -> int:
        return hhmm_to_minutes(self.start)

    @property
    def end_minute(self) -> int:
        return hhmm_to_minutes(self.end)

    def validate(self) -> None:
        PeakWindow(start=self.start, end=self.end).validate(allow_wraparound=False)
        if not np.isfinite(float(self.price)):
            raise ValueError("energy price must be finite")


@dataclass(frozen=True)
class SeasonalTariffBlock:
    name: str
    calendar: CalendarMask
    demand_windows: tuple[DemandWindowRule, ...]
    energy_price_rules: tuple[EnergyPriceRule, ...]
    default_energy_price: float

    def validate(self) -> None:
        self.calendar.validate()
        if not self.energy_price_rules:
            raise ValueError("seasonal tariff block requires at least one energy price rule")
        if not np.isfinite(float(self.default_energy_price)):
            raise ValueError("default_energy_price must be finite")
        for window in self.demand_windows:
            window.validate()
        for rule in self.energy_price_rules:
            rule.validate()


@dataclass(frozen=True)
class TariffIR:
    timezone: str
    dt_minutes: int
    demand_charge_rate_kw: float
    weekmask: Weekmask = "MON_FRI"
    peak_window: PeakWindow | None = None
    energy_price_offpeak: float | None = None
    energy_price_peak: float | None = None
    demand_charge_mode: DemandChargeMode = "PEAK_WINDOW_MONTHLY_MAX"
    seasonal_blocks: tuple[SeasonalTariffBlock, ...] = ()
    family_id: str | None = None
    variant_id: int | None = None
    label: str | None = None

    def validate_paper_invariants(self) -> None:
        if self.dt_minutes != 15:
            raise ValueError("paper setting requires dt_minutes=15")
        if self.demand_charge_mode != "PEAK_WINDOW_MONTHLY_MAX":
            raise ValueError("paper setting requires PEAK_WINDOW_MONTHLY_MAX")
        if self.seasonal_blocks:
            seen_months: set[int] = set()
            for block in self.seasonal_blocks:
                block.validate()
                months = set(int(m) for m in block.calendar.months)
                overlap = seen_months & months
                if overlap:
                    raise ValueError(f"seasonal blocks must have disjoint months; overlap={sorted(overlap)}")
                seen_months |= months
            return
        if self.peak_window is None:
            raise ValueError("legacy tariff requires peak_window")
        if self.energy_price_offpeak is None or self.energy_price_peak is None:
            raise ValueError("legacy tariff requires energy_price_offpeak and energy_price_peak")
        self.peak_window.validate(allow_wraparound=False)
        if self.weekmask not in ("MON_FRI", "ALL_DAYS"):
            raise ValueError(f"invalid weekmask: {self.weekmask}")


def make_tariff(
    *,
    timezone: str = "UTC",
    dt_minutes: int = 15,
    weekmask: Weekmask = "MON_FRI",
    peak_start: str,
    peak_end: str,
    energy_price_offpeak: float,
    energy_price_peak: float,
    demand_charge_rate_kw: float,
) -> TariffIR:
    tariff = TariffIR(
        timezone=timezone,
        dt_minutes=dt_minutes,
        weekmask=weekmask,
        peak_window=PeakWindow(start=peak_start, end=peak_end),
        energy_price_offpeak=float(energy_price_offpeak),
        energy_price_peak=float(energy_price_peak),
        demand_charge_rate_kw=float(demand_charge_rate_kw),
    )
    tariff.validate_paper_invariants()
    return tariff


@dataclass(frozen=True)
class CompiledTariff:
    is_peak_window: np.ndarray
    energy_price: np.ndarray
    minutes_to_window_start: np.ndarray
    minutes_to_window_end: np.ndarray
    active_schedule_id: np.ndarray | None = None
    is_super_offpeak: np.ndarray | None = None
    is_midpeak: np.ndarray | None = None


def _is_weekday_mon_fri(dow: int) -> bool:
    return int(dow) in (0, 1, 2, 3, 4)


def _is_weekmask_active(weekmask: Weekmask, dows: np.ndarray) -> np.ndarray:
    if weekmask == "ALL_DAYS":
        return np.ones_like(dows, dtype=bool)
    return np.array([_is_weekday_mon_fri(int(d)) for d in dows], dtype=bool)


def _derive_price_level_flags(energy_price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    price = np.asarray(energy_price, dtype=float)
    levels = np.unique(price[np.isfinite(price)])
    if levels.size < 3:
        return np.zeros(price.shape, dtype=int), np.zeros(price.shape, dtype=int)
    levels = np.sort(levels)
    super_level = float(levels[0])
    mid_level = float(levels[1])
    return (price == super_level).astype(int), (price == mid_level).astype(int)


def compile_tariff(
    tariff: TariffIR,
    timestamps_utc: pd.DatetimeIndex,
    *,
    energy_price_mode: Literal["TOU", "CONSTANT"] = "TOU",
) -> CompiledTariff:
    tariff.validate_paper_invariants()

    tz = ensure_zoneinfo(tariff.timezone)
    if timestamps_utc.tz is None:
        raise ValueError("timestamps_utc must be timezone-aware")
    ts_local = timestamps_utc.tz_convert(tz)

    minutes = (ts_local.hour * 60 + ts_local.minute).to_numpy(dtype=int)
    dows = ts_local.dayofweek.to_numpy(dtype=int)
    months = ts_local.month.to_numpy(dtype=int)

    mode = str(energy_price_mode).upper().strip()
    if mode not in {"TOU", "CONSTANT"}:
        raise ValueError(f"invalid energy_price_mode: {energy_price_mode!r}")

    if tariff.seasonal_blocks:
        is_peak_window = np.zeros(minutes.shape, dtype=int)
        energy_price = np.full(minutes.shape, np.nan, dtype=float)
        minutes_to_window_start = np.full(minutes.shape, 1440, dtype=int)
        minutes_to_window_end = np.full(minutes.shape, 1440, dtype=int)
        active_schedule_id = np.full(minutes.shape, -1, dtype=int)

        for block_idx, block in enumerate(tariff.seasonal_blocks):
            month_mask = np.isin(months, np.asarray(block.calendar.months, dtype=int))
            if not np.any(month_mask):
                continue

            idx = np.where(month_mask)[0]
            active_schedule_id[idx] = int(block_idx)

            day_active = _is_weekmask_active(block.calendar.weekmask, dows[idx])
            energy_price[idx] = float(block.default_energy_price)

            active_idx = idx[day_active]
            if active_idx.size == 0:
                continue

            block_minutes = minutes[active_idx]

            if mode == "TOU":
                for rule in block.energy_price_rules:
                    rule_mask = (block_minutes >= rule.start_minute) & (block_minutes < rule.end_minute)
                    if np.any(rule_mask):
                        energy_price[active_idx[rule_mask]] = float(rule.price)
            else:
                energy_price[active_idx] = float(block.default_energy_price)

            if block.demand_windows:
                start_dist = np.full(block_minutes.shape, 1440, dtype=int)
                end_dist = np.full(block_minutes.shape, 1440, dtype=int)
                in_any_window = np.zeros(block_minutes.shape, dtype=bool)
                for window in block.demand_windows:
                    win_mask = (block_minutes >= window.start_minute) & (block_minutes < window.end_minute)
                    in_any_window |= win_mask
                    start_dist = np.minimum(
                        start_dist,
                        np.array([minutes_until(window.start_minute, int(m)) for m in block_minutes], dtype=int),
                    )
                    end_dist = np.minimum(
                        end_dist,
                        np.array([minutes_until(window.end_minute, int(m)) for m in block_minutes], dtype=int),
                    )
                is_peak_window[active_idx[in_any_window]] = 1
                minutes_to_window_start[active_idx] = start_dist
                minutes_to_window_end[active_idx] = end_dist

        if np.isnan(energy_price).any():
            nan_count = int(np.isnan(energy_price).sum())
            raise ValueError(
                f"compile_tariff produced {nan_count} NaN energy prices. "
                "Seasonal blocks do not cover all timestamps. "
                "Check that CalendarMask months are exhaustive and non-overlapping."
            )

        is_super_offpeak, is_midpeak = _derive_price_level_flags(energy_price)
        return CompiledTariff(
            is_peak_window=is_peak_window,
            energy_price=energy_price.astype(float),
            minutes_to_window_start=minutes_to_window_start,
            minutes_to_window_end=minutes_to_window_end,
            active_schedule_id=active_schedule_id,
            is_super_offpeak=is_super_offpeak,
            is_midpeak=is_midpeak,
        )

    if tariff.peak_window is None:
        raise ValueError("legacy tariff requires peak_window")

    start = int(tariff.peak_window.start_minute)
    end = int(tariff.peak_window.end_minute)
    if end <= start:
        raise ValueError("wrap-around not allowed in paper suite")

    in_window = (minutes >= start) & (minutes < end)

    is_ok_day = None
    if tariff.weekmask == "MON_FRI":
        is_ok_day = _is_weekmask_active(tariff.weekmask, dows)
        in_window = in_window & is_ok_day

    is_peak_window = in_window.astype(int)

    if mode == "CONSTANT":
        energy_price = np.full_like(minutes, float(tariff.energy_price_offpeak), dtype=float)
    else:
        # price: peak within window, else off-peak
        energy_price = np.where(in_window, float(tariff.energy_price_peak), float(tariff.energy_price_offpeak)).astype(float)

    minutes_to_window_start = np.array([minutes_until(start, int(m)) for m in minutes], dtype=int)
    minutes_to_window_end = np.array([minutes_until(end, int(m)) for m in minutes], dtype=int)
    if is_ok_day is not None:
        # Setze auf 1440 (invalid), wenn es KEIN ok_day ist (z.B. Wochenende)
        minutes_to_window_start = np.where(is_ok_day, minutes_to_window_start, 1440)
        minutes_to_window_end = np.where(is_ok_day, minutes_to_window_end, 1440)

    is_super_offpeak, is_midpeak = _derive_price_level_flags(energy_price)

    return CompiledTariff(
        is_peak_window=is_peak_window,
        energy_price=energy_price,
        minutes_to_window_start=minutes_to_window_start,
        minutes_to_window_end=minutes_to_window_end,
        active_schedule_id=np.zeros(minutes.shape, dtype=int),
        is_super_offpeak=is_super_offpeak,
        is_midpeak=is_midpeak,
    )


def make_experiment_tariff_suite(*, timezone: str = "UTC", weekmask: Weekmask = "MON_FRI") -> dict[str, TariffIR]:
    demand = 15.0
    off = 0.10
    peak = 0.20

    def _legacy_peak(label: str, variant_id: int, peak_start: str, peak_end: str, *, flat: bool = False) -> TariffIR:
        return TariffIR(
            timezone=timezone,
            dt_minutes=15,
            weekmask=weekmask,
            peak_window=PeakWindow(start=peak_start, end=peak_end),
            energy_price_offpeak=off,
            energy_price_peak=(off if flat else peak),
            demand_charge_rate_kw=demand,
            family_id="peak_shift",
            variant_id=int(variant_id),
            label=label,
        )

    def _seasonal(label: str, variant_id: int, windows: dict[str, tuple[str, str]]) -> TariffIR:
        blocks = (
            SeasonalTariffBlock(
                name="winter",
                calendar=CalendarMask(months=(1, 2, 11, 12), weekmask=weekmask),
                demand_windows=(DemandWindowRule(*windows["winter"]),),
                energy_price_rules=(EnergyPriceRule(*windows["winter"], price=peak, label="peak"),),
                default_energy_price=off,
            ),
            SeasonalTariffBlock(
                name="shoulder",
                calendar=CalendarMask(months=(3, 4, 5, 10), weekmask=weekmask),
                demand_windows=(DemandWindowRule(*windows["shoulder"]),),
                energy_price_rules=(EnergyPriceRule(*windows["shoulder"], price=peak, label="peak"),),
                default_energy_price=off,
            ),
            SeasonalTariffBlock(
                name="summer",
                calendar=CalendarMask(months=(6, 7, 8, 9), weekmask=weekmask),
                demand_windows=(DemandWindowRule(*windows["summer"]),),
                energy_price_rules=(EnergyPriceRule(*windows["summer"], price=peak, label="peak"),),
                default_energy_price=off,
            ),
        )
        return TariffIR(
            timezone=timezone,
            dt_minutes=15,
            demand_charge_rate_kw=demand,
            seasonal_blocks=blocks,
            family_id="seasonal_window",
            variant_id=int(variant_id),
            label=label,
        )

    def _tiered(label: str, variant_id: int, *, super_window: tuple[str, str], peak_window: tuple[str, str], super_price: float, off_price: float, peak_price: float) -> TariffIR:
        blocks = (
            SeasonalTariffBlock(
                name="all_year",
                calendar=CalendarMask(months=tuple(range(1, 13)), weekmask=weekmask),
                demand_windows=(DemandWindowRule(*peak_window),),
                energy_price_rules=(
                    EnergyPriceRule(super_window[0], super_window[1], price=super_price, label="super_offpeak"),
                    EnergyPriceRule(peak_window[0], peak_window[1], price=peak_price, label="peak"),
                ),
                default_energy_price=off_price,
            ),
        )
        return TariffIR(
            timezone=timezone,
            dt_minutes=15,
            demand_charge_rate_kw=demand,
            seasonal_blocks=blocks,
            family_id="tiered_tou",
            variant_id=int(variant_id),
            label=label,
        )

    suite = {
        "P1": _legacy_peak("P1", 1, "07:00", "11:00"),
        "P2": _legacy_peak("P2", 2, "11:00", "15:00"),
        "P3": _legacy_peak("P3", 3, "15:00", "19:00"),
        "P4": _legacy_peak("P4", 4, "17:00", "21:00"),
        "P3_flat": _legacy_peak("P3_flat", 3, "15:00", "19:00", flat=True),
        "S1": _seasonal("S1", 1, {"winter": ("07:00", "11:00"), "shoulder": ("12:00", "16:00"), "summer": ("16:00", "20:00")}),
        "S2": _seasonal("S2", 2, {"winter": ("06:00", "10:00"), "shoulder": ("11:00", "15:00"), "summer": ("15:00", "19:00")}),
        "S3": _seasonal("S3", 3, {"winter": ("08:00", "12:00"), "shoulder": ("13:00", "17:00"), "summer": ("14:00", "18:00")}),
        "S4": _seasonal("S4", 4, {"winter": ("09:00", "13:00"), "shoulder": ("14:00", "18:00"), "summer": ("17:00", "21:00")}),
        "T1": _tiered("T1", 1, super_window=("00:00", "06:00"), peak_window=("16:00", "20:00"), super_price=0.10, off_price=0.18, peak_price=0.30),
        "T2": _tiered("T2", 2, super_window=("00:00", "05:00"), peak_window=("15:00", "21:00"), super_price=0.09, off_price=0.17, peak_price=0.32),
        "T3": _tiered("T3", 3, super_window=("00:00", "06:00"), peak_window=("14:00", "19:00"), super_price=0.11, off_price=0.19, peak_price=0.29),
        "T4": _tiered("T4", 4, super_window=("00:00", "07:00"), peak_window=("17:00", "22:00"), super_price=0.08, off_price=0.16, peak_price=0.31),
    }
    for tariff in suite.values():
        tariff.validate_paper_invariants()
    return suite


def tariff_family_from_name(name: str) -> str:
    label = str(name)
    if label.startswith("P"):
        return "peak_shift"
    if label.startswith("S"):
        return "seasonal_window"
    if label.startswith("T"):
        return "tiered_tou"
    if label in {"A", "B", "C", "C_flat"}:
        return "legacy"
    raise ValueError(f"cannot infer tariff family from name: {name}")


def make_paper_suite(*, timezone: str = "UTC", weekmask: Weekmask = "MON_FRI") -> dict[str, TariffIR]:
    """A/B/C/C_flat legacy synthetic suite (single-family peak-shift only)."""

    # Values are intentionally generic; only window shifts matter.
    off = 0.10
    peak = 0.20
    demand = 15.0

    A = make_tariff(
        timezone=timezone,
        weekmask=weekmask,
        peak_start="12:00",
        peak_end="16:00",
        energy_price_offpeak=off,
        energy_price_peak=peak,
        demand_charge_rate_kw=demand,
    )
    B = make_tariff(
        timezone=timezone,
        weekmask=weekmask,
        peak_start="17:00",
        peak_end="21:00",
        energy_price_offpeak=off,
        energy_price_peak=peak,
        demand_charge_rate_kw=demand,
    )
    C = make_tariff(
        timezone=timezone,
        weekmask=weekmask,
        peak_start="07:00",
        peak_end="11:00",
        energy_price_offpeak=off,
        energy_price_peak=peak,
        demand_charge_rate_kw=demand,
    )

    # Confound control: flat energy price, same window
    C_flat = TariffIR(
        timezone=timezone,
        dt_minutes=15,
        weekmask=weekmask,
        peak_window=PeakWindow(start="07:00", end="11:00"),
        energy_price_offpeak=off,
        energy_price_peak=off,
        demand_charge_rate_kw=demand,
    )
    C_flat.validate_paper_invariants()

    return {"A": A, "B": B, "C": C, "C_flat": C_flat}


# -----------------------------
# billing
# -----------------------------


@dataclass(frozen=True)
class BillBreakdown:
    bill_total: float
    energy_cost: float
    demand_cost: float
    peak_kw: float


def compute_monthly_bill(
    *,
    grid_kw: np.ndarray,
    energy_price_per_kwh: np.ndarray,
    is_peak_window: np.ndarray,
    demand_charge_rate_kw: float,
    dt_hours: float,
) -> BillBreakdown:
    grid_kw = np.asarray(grid_kw, dtype=float)
    price = np.asarray(energy_price_per_kwh, dtype=float)
    peak = np.asarray(is_peak_window, dtype=int)
    if not (grid_kw.shape == price.shape == peak.shape):
        raise ValueError("grid_kw, energy_price_per_kwh, is_peak_window must have same shape")

    grid_kwh = grid_kw * float(dt_hours)
    energy_cost = float(np.sum(grid_kwh * price))

    # PEAK_WINDOW_MONTHLY_MAX
    if int(np.max(peak)) == 0:
        peak_kw = 0.0
    else:
        peak_kw = float(np.max(grid_kw[peak.astype(bool)]))

    demand_cost = float(demand_charge_rate_kw) * float(peak_kw)
    bill_total = float(energy_cost + demand_cost)

    return BillBreakdown(bill_total=bill_total, energy_cost=energy_cost, demand_cost=demand_cost, peak_kw=peak_kw)


# -----------------------------
# env
# -----------------------------


@dataclass(frozen=True)
class BatteryParams:
    E_max_kwh: float
    P_max_kw: float
    eta_charge: float
    eta_discharge: float
    dt_hours: float

    def validate(self) -> None:
        if self.E_max_kwh <= 0:
            raise ValueError("E_max_kwh must be > 0")
        if self.P_max_kw <= 0:
            raise ValueError("P_max_kw must be > 0")
        if not (0 < self.eta_charge <= 1.0):
            raise ValueError("eta_charge must be in (0,1]")
        if not (0 < self.eta_discharge <= 1.0):
            raise ValueError("eta_discharge must be in (0,1]")
        if self.dt_hours <= 0:
            raise ValueError("dt_hours must be > 0")


@dataclass(frozen=True)
class EnvState:
    E_kwh: float
    current_max_peak_kw: float


@dataclass(frozen=True)
class EnvConfig:
    allow_grid_export: bool = False


def clip_action_to_soc_bounds(
    *,
    state: EnvState | None = None,
    E_kwh: float | None = None,
    P_batt_kw_raw: float | None = None,
    P_batt_kw: float | None = None,
    battery: BatteryParams | None = None,
    params: BatteryParams | None = None,
) -> float:
    """Clip an action to SOC constraints.

    Backward-compatible call styles:
    - clip_action_to_soc_bounds(state=..., P_batt_kw_raw=..., battery=...)
    - clip_action_to_soc_bounds(E_kwh=..., P_batt_kw=..., params=...)
    """

    if battery is None:
        battery = params
    if battery is None:
        raise TypeError("battery/params is required")
    battery.validate()

    if state is None:
        if E_kwh is None:
            raise TypeError("state or E_kwh is required")
        state = EnvState(E_kwh=float(E_kwh), current_max_peak_kw=0.0)

    if P_batt_kw_raw is None:
        if P_batt_kw is None:
            raise TypeError("P_batt_kw_raw or P_batt_kw is required")
        P_batt_kw_raw = float(P_batt_kw)

    E = float(state.E_kwh)
    dt = float(battery.dt_hours)
    Pmax = float(battery.P_max_kw)
    eta_c = float(battery.eta_charge)
    eta_d = float(battery.eta_discharge)

    P_charge_max = min(Pmax, (float(battery.E_max_kwh) - E) / (eta_c * dt))
    P_discharge_min = max(-Pmax, -(E * eta_d) / dt)

    return float(np.clip(float(P_batt_kw_raw), float(P_discharge_min), float(P_charge_max)))


@dataclass(frozen=True)
class StepResult:
    next_state: EnvState
    P_batt_kw_safe: float
    grid_kw: float
    grid_kw_clamped: float


def step_env(
    *,
    state: EnvState,
    load_kw: float,
    P_batt_kw_raw: float,
    is_peak_window: int,
    battery: BatteryParams,
    env: EnvConfig,
) -> StepResult:
    battery.validate()

    P_safe = clip_action_to_soc_bounds(state=state, P_batt_kw_raw=float(P_batt_kw_raw), battery=battery)

    # Sign convention: + charge, - discharge
    grid_kw = float(load_kw) + float(P_safe)
    grid_kw_clamped = float(grid_kw)
    if not env.allow_grid_export:
        grid_kw_clamped = max(0.0, float(grid_kw))

    E = float(state.E_kwh)
    dt = float(battery.dt_hours)
    eta_c = float(battery.eta_charge)
    eta_d = float(battery.eta_discharge)

    if P_safe >= 0:
        E_next = E + eta_c * float(P_safe) * dt
    else:
        E_next = E - (1.0 / eta_d) * float(-P_safe) * dt

    E_next = float(np.clip(E_next, 0.0, float(battery.E_max_kwh)))

    current_max_peak_kw = float(state.current_max_peak_kw)
    if int(is_peak_window) == 1:
        current_max_peak_kw = max(current_max_peak_kw, float(grid_kw_clamped))

    return StepResult(
        next_state=EnvState(E_kwh=E_next, current_max_peak_kw=current_max_peak_kw),
        P_batt_kw_safe=float(P_safe),
        grid_kw=float(grid_kw),
        grid_kw_clamped=float(grid_kw_clamped),
    )


# -----------------------------
# teacher
# -----------------------------


@dataclass(frozen=True)
class SolverLog:
    requested_solver: str
    solver_used: str
    used_fallback_solver: bool
    status: str
    solve_time_sec: float
    extra: dict[str, Any]


@dataclass(frozen=True)
class TeacherConfig:
    solver: str = "GUROBI"
    allow_solver_fallback: bool = False
    fallback_solver_order: tuple[str, ...] = ("OSQP", "ECOS", "SCS")
    time_limit_sec: float | None = None
    lambda_batt_power: float = 0.0
    gurobi_threads: int = 1


@dataclass(frozen=True)
class ForecastNoiseConfig:
    sigma_rel: float = 0.0
    rho: float = 0.0
    seed: int = 0


def apply_ar1_forecast_noise(
    load_kw: np.ndarray,
    *,
    sigma_rel: float,
    rho: float,
    seed: int,
) -> np.ndarray:
    load = np.asarray(load_kw, dtype=float)
    if load.ndim != 1:
        raise ValueError("load_kw must be 1D")
    sigma = float(sigma_rel)
    corr = float(rho)
    if sigma <= 0.0:
        return load.copy()
    if abs(corr) >= 1.0:
        raise ValueError("rho must be in (-1, 1)")

    scale = max(float(np.mean(np.abs(load))), 1e-6)
    innovation_std = sigma * scale * np.sqrt(max(1.0 - corr * corr, 1e-12))
    rng = np.random.default_rng(int(seed))
    noise = np.zeros_like(load, dtype=float)
    for idx in range(load.shape[0]):
        innovation = float(rng.normal(loc=0.0, scale=innovation_std))
        if idx == 0:
            noise[idx] = innovation
        else:
            noise[idx] = corr * noise[idx - 1] + innovation
    return np.maximum(load + noise, 0.0)


@dataclass(frozen=True)
class MPCResult:
    P_batt_kw: np.ndarray
    E_kwh: np.ndarray
    monthly_peak_kw: float
    solver_log: SolverLog


def _require_cvxpy() -> Any:
    if cp is None:
        raise RuntimeError("cvxpy is required for the teacher MPC. Install it via `pip install cvxpy`.")
    return cp


def _select_solver(cfg: TeacherConfig) -> tuple[str, bool]:
    _cp = _require_cvxpy()
    installed = set(_cp.installed_solvers())
    requested = cfg.solver
    if requested in installed:
        return requested, False
    if not cfg.allow_solver_fallback:
        raise RuntimeError(f"Requested solver {requested} not available; installed={sorted(installed)}")
    for s in cfg.fallback_solver_order:
        if s in installed:
            return s, True
    raise RuntimeError(f"No fallback solver available; installed={sorted(installed)}")


def solve_day_ahead_mpc(
    *,
    load_kw: np.ndarray,
    energy_price_per_kwh: np.ndarray,
    is_peak_window: np.ndarray,
    demand_charge_rate_kw: float,
    current_E_kwh: float,
    current_max_peak_kw: float,
    battery: BatteryParams,
    allow_grid_export: bool,
    teacher: TeacherConfig,
) -> MPCResult:
    _cp = _require_cvxpy()

    load_kw = np.asarray(load_kw, dtype=float)
    price = np.asarray(energy_price_per_kwh, dtype=float)
    peak_mask = np.asarray(is_peak_window).astype(int)

    if load_kw.ndim != 1:
        raise ValueError("load_kw must be 1D")
    if price.shape != load_kw.shape or peak_mask.shape != load_kw.shape:
        raise ValueError("load_kw, energy_price_per_kwh, is_peak_window must have same shape")

    T = int(load_kw.shape[0])
    battery.validate()

    P_charge = _cp.Variable(T, nonneg=True)
    P_discharge = _cp.Variable(T, nonneg=True)
    P_net = P_charge - P_discharge
    E = _cp.Variable(T + 1)

    grid_kw = load_kw + P_net

    constraints: list[Any] = []
    constraints += [E[0] == float(current_E_kwh)]
    constraints += [E >= 0.0, E <= float(battery.E_max_kwh)]
    constraints += [P_charge <= float(battery.P_max_kw)]
    constraints += [P_discharge <= float(battery.P_max_kw)]

    dt = float(battery.dt_hours)
    eta_c = float(battery.eta_charge)
    eta_d = float(battery.eta_discharge)
    # Vectorised SOC dynamics: E[1:] == E[:-1] + eta_c*P_charge*dt - (1/eta_d)*P_discharge*dt
    constraints += [
        E[1:] == E[:-1] + eta_c * P_charge * dt - (1.0 / eta_d) * P_discharge * dt
    ]

    if not allow_grid_export:
        constraints += [grid_kw >= 0.0]

    monthly_peak_kw = _cp.Variable()
    constraints += [monthly_peak_kw >= float(current_max_peak_kw)]
    constraints += [monthly_peak_kw >= 0.0]
    # Vectorised peak constraint: only at timesteps where peak_mask==1
    peak_idx = np.flatnonzero(peak_mask)
    if peak_idx.size > 0:
        constraints += [monthly_peak_kw >= grid_kw[peak_idx]]

    energy_cost = _cp.sum(_cp.multiply(grid_kw * dt, price))
    demand_cost = float(demand_charge_rate_kw) * monthly_peak_kw

    eps_qp = 1e-9
    lam = float(teacher.lambda_batt_power)
    reg = max(lam, eps_qp) * (_cp.sum_squares(P_charge) + _cp.sum_squares(P_discharge))

    prob = _cp.Problem(_cp.Minimize(energy_cost + demand_cost + reg), constraints)

    solver_used, used_fallback = _select_solver(teacher)

    solve_kwargs: dict[str, Any] = {"solver": solver_used, "warm_start": True}
    if solver_used == "GUROBI":
        solve_kwargs["Threads"] = max(1, int(getattr(teacher, 'gurobi_threads', 0) or 1))
        if teacher.time_limit_sec is not None:
            solve_kwargs["TimeLimit"] = float(teacher.time_limit_sec)
    elif solver_used == "OSQP":
        if teacher.time_limit_sec is not None:
            solve_kwargs["time_limit"] = float(teacher.time_limit_sec)

    t0 = perf_counter()
    prob.solve(**solve_kwargs)
    t1 = perf_counter()

    if (
        P_charge.value is None
        or P_discharge.value is None
        or E.value is None
        or monthly_peak_kw.value is None
    ):
        raise RuntimeError(f"MPC solve failed: status={prob.status}")

    return MPCResult(
        P_batt_kw=np.asarray((P_charge.value - P_discharge.value), dtype=float),
        E_kwh=np.asarray(E.value, dtype=float),
        monthly_peak_kw=float(monthly_peak_kw.value),
        solver_log=SolverLog(
            requested_solver=teacher.solver,
            solver_used=str(solver_used),
            used_fallback_solver=bool(used_fallback),
            status=str(prob.status),
            solve_time_sec=float(t1 - t0),
            extra={"cvxpy_status": str(prob.status)},
        ),
    )


@dataclass(frozen=True)
class TeacherRollout:
    P_batt_kw: np.ndarray
    E_kwh: np.ndarray
    current_max_peak_kw: np.ndarray
    grid_kw: np.ndarray
    grid_kw_clamped: np.ndarray
    monthly_peak_kw_end: float


def run_teacher_receding_horizon(
    *,
    load_kw: np.ndarray,
    energy_price_per_kwh: np.ndarray,
    is_peak_window: np.ndarray,
    demand_charge_rate_kw: float,
    battery: BatteryParams,
    allow_grid_export: bool,
    teacher: TeacherConfig,
    horizon_steps: int,
    initial_E_kwh: float,
    initial_max_peak_kw: float,
    forecast_load_kw: np.ndarray | None = None,
) -> tuple[TeacherRollout, list[MPCResult]]:
    load_kw = np.asarray(load_kw, dtype=float)
    price = np.asarray(energy_price_per_kwh, dtype=float)
    peak = np.asarray(is_peak_window, dtype=int)
    if not (load_kw.shape == price.shape == peak.shape):
        raise ValueError("load_kw, energy_price_per_kwh, is_peak_window must have same shape")
    forecast_load = load_kw if forecast_load_kw is None else np.asarray(forecast_load_kw, dtype=float)
    if forecast_load.shape != load_kw.shape:
        raise ValueError("forecast_load_kw must match load_kw shape")

    T = int(load_kw.shape[0])
    horizon_steps = int(horizon_steps)
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")

    env = EnvConfig(allow_grid_export=bool(allow_grid_export))
    state = EnvState(E_kwh=float(initial_E_kwh), current_max_peak_kw=float(initial_max_peak_kw))

    P_hist = np.zeros(T, dtype=float)
    E_hist = np.zeros(T + 1, dtype=float)
    peak_hist = np.zeros(T + 1, dtype=float)
    grid_hist = np.zeros(T, dtype=float)
    gridc_hist = np.zeros(T, dtype=float)
    E_hist[0] = float(state.E_kwh)
    peak_hist[0] = float(state.current_max_peak_kw)

    solves: list[MPCResult] = []

    for t in range(T):
        end = min(T, t + horizon_steps)
        mpc = solve_day_ahead_mpc(
            load_kw=forecast_load[t:end],
            energy_price_per_kwh=price[t:end],
            is_peak_window=peak[t:end],
            demand_charge_rate_kw=float(demand_charge_rate_kw),
            current_E_kwh=float(state.E_kwh),
            current_max_peak_kw=float(state.current_max_peak_kw),
            battery=battery,
            allow_grid_export=bool(allow_grid_export),
            teacher=teacher,
        )
        solves.append(mpc)
        action = float(mpc.P_batt_kw[0])

        step = step_env(
            state=state,
            load_kw=float(load_kw[t]),
            P_batt_kw_raw=action,
            is_peak_window=int(peak[t]),
            battery=battery,
            env=env,
        )

        P_hist[t] = float(step.P_batt_kw_safe)
        grid_hist[t] = float(step.grid_kw)
        gridc_hist[t] = float(step.grid_kw_clamped)

        state = step.next_state
        E_hist[t + 1] = float(state.E_kwh)
        peak_hist[t + 1] = float(state.current_max_peak_kw)

    return (
        TeacherRollout(
            P_batt_kw=P_hist,
            E_kwh=E_hist,
            current_max_peak_kw=peak_hist,
            grid_kw=grid_hist,
            grid_kw_clamped=gridc_hist,
            monthly_peak_kw_end=float(state.current_max_peak_kw),
        ),
        solves,
    )
