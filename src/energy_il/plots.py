from __future__ import annotations

"""Plotting helpers.

Kept as a separate module to keep `core.py` and `tail.py` focused.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TariffPlotData:
    energy_price: np.ndarray
    is_peak_window: np.ndarray
    minutes_to_window_start: np.ndarray
    minutes_to_window_end: np.ndarray


def make_tariff_plot_data(compiled) -> TariffPlotData:
    return TariffPlotData(
        energy_price=np.asarray(compiled.energy_price, dtype=float),
        is_peak_window=np.asarray(compiled.is_peak_window, dtype=int),
        minutes_to_window_start=np.asarray(compiled.minutes_to_window_start, dtype=int),
        minutes_to_window_end=np.asarray(compiled.minutes_to_window_end, dtype=int),
    )


def plot_tariff(ax, plot_data: TariffPlotData, *, title: str):
    import matplotlib.pyplot as plt  # local import

    t = np.arange(plot_data.energy_price.shape[0])
    ax.plot(t, plot_data.energy_price, label="energy price")
    ax.fill_between(t, 0, plot_data.energy_price.max(), where=plot_data.is_peak_window.astype(bool), alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("$/kWh")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")


def plot_timeseries(ax, *, load_kw: np.ndarray, grid_kw: np.ndarray, P_batt_kw: np.ndarray, title: str):
    t = np.arange(len(load_kw))
    ax.plot(t, load_kw, label="load_kw", alpha=0.8)
    ax.plot(t, grid_kw, label="grid_kw", alpha=0.8)
    ax.plot(t, P_batt_kw, label="P_batt_kw", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")


def plot_bar(ax, *, labels: list[str], values: list[float], title: str, ylabel: str):
    import matplotlib.pyplot as plt  # local import

    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
