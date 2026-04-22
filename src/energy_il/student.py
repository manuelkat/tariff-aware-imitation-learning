from __future__ import annotations

"""Student models + training (BC + DAgger-lite).

This module consolidates:
- features
- torch model
- BC training
- DAgger-lite training
- evaluation rollouts

API is designed to be called from `tail.py` and to satisfy the existing tests.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from . import core


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the student models. Install it via `pip install torch`.")
    return torch


@dataclass(frozen=True)
class FeatureConfig:
    include_time: bool = True
    include_tariff: bool = True
    include_state: bool = True
    include_load: bool = True
    include_peak_flag: bool = True
    include_tariff_price: bool = True
    include_minutes_to_boundary: bool = True


@dataclass(frozen=True)
class FeatureScales:
    load_kw_scale: float
    price_scale: float


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return _require_torch().device("cpu")


def make_features(
    *,
    load_kw: np.ndarray,
    state_E_kwh: np.ndarray,
    state_current_max_peak_kw: np.ndarray,
    compiled: core.CompiledTariff,
    time_features: np.ndarray,
    cfg: FeatureConfig,
    battery_E_max_kwh: float,
    scales: FeatureScales,
) -> np.ndarray:
    load_kw = np.asarray(load_kw, dtype=float)
    E = np.asarray(state_E_kwh, dtype=float)
    peak = np.asarray(state_current_max_peak_kw, dtype=float)
    if not (load_kw.shape == E.shape == peak.shape):
        raise ValueError("load_kw/state arrays must match")

    load_scale = max(float(scales.load_kw_scale), 1e-6)
    price_scale = max(float(scales.price_scale), 1e-6)
    E_max = max(float(battery_E_max_kwh), 1e-6)

    feats: list[np.ndarray] = []

    if cfg.include_load:
        feats.append((load_kw / load_scale).reshape(-1, 1))

    if cfg.include_state:
        feats.append((E / E_max).reshape(-1, 1))
        feats.append((peak / load_scale).reshape(-1, 1))

    if cfg.include_tariff:
        if cfg.include_peak_flag:
            feats.append(np.asarray(compiled.is_peak_window, dtype=float).reshape(-1, 1))
        if cfg.include_tariff_price:
            feats.append((np.asarray(compiled.energy_price, dtype=float) / price_scale).reshape(-1, 1))
        if cfg.include_minutes_to_boundary:
            feats.append((np.asarray(compiled.minutes_to_window_start, dtype=float) / 1440.0).reshape(-1, 1))
            feats.append((np.asarray(compiled.minutes_to_window_end, dtype=float) / 1440.0).reshape(-1, 1))

    if cfg.include_time:
        # time_features: [month, minute_of_day, dow, holiday, month_progress]
        tf = np.asarray(time_features, dtype=float)
        if tf.ndim != 2 or tf.shape[0] != load_kw.shape[0]:
            raise ValueError("time_features shape mismatch")
        # Normalize to keep magnitudes stable.
        month = (tf[:, 0:1] / 12.0)
        minute = (tf[:, 1:2] / 1440.0)
        dow = (tf[:, 2:3] / 6.0)
        hol = tf[:, 3:4]
        if tf.shape[1] >= 5:
            month_progress = tf[:, 4:5]
            feats.append(np.concatenate([month, minute, dow, hol, month_progress], axis=1))
        else:
            feats.append(np.concatenate([month, minute, dow, hol], axis=1))

    X = np.concatenate(feats, axis=1)
    return X.astype(np.float32)


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, cfg: MLPConfig):
        _require_torch()
        super().__init__()

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    seed: int = 0
    weight_decay: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    cost_eval_every: int = 0
    cost_patience_evals: int = 0
    device: str = "cpu"
    restore_best_checkpoint_by: str = "val_mse"


@dataclass(frozen=True)
class FitResult:
    model: Any
    train_loss: float
    loss_history: list[float] | None = None
    val_loss_history: list[float] | None = None
    cost_eval_history: list[dict[str, Any]] | None = None
    best_epoch: int | None = None
    restored_by: str | None = None
    best_val_loss: float | None = None
    best_cost: float | None = None
    total_epochs: int | None = None
    best_val_mse_state: dict | None = None
    best_val_mse_epoch: int | None = None


def _iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], y[sl]


def fit_bc_policy(
    *,
    X: np.ndarray,
    y_action_kw: np.ndarray,
    train: TrainConfig,
    model_cfg: MLPConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cost_eval_fn: Any | None = None,
) -> FitResult:
    _t = _require_torch()
    rng = np.random.default_rng(int(train.seed))
    _t.manual_seed(int(train.seed))
    requested_device = str(train.device).strip().lower()
    if requested_device.startswith("cuda") and not _t.cuda.is_available():
        requested_device = "cpu"
    device = _t.device(requested_device)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_action_kw, dtype=np.float32).reshape(-1, 1)

    model = MLPPolicy(input_dim=X.shape[1], cfg=model_cfg).to(device)
    opt = _t.optim.Adam(model.parameters(), lr=float(train.lr), weight_decay=float(train.weight_decay))
    loss_fn = _t.nn.MSELoss()

    model.train()
    last_loss = 0.0
    loss_history: list[float] = []
    val_history: list[float] = []
    best_val = float("inf")
    best_val_state: dict[str, Any] | None = None
    best_val_epoch: int = 0
    patience = max(0, int(train.early_stopping_patience))
    min_delta = float(train.early_stopping_min_delta)
    no_improve = 0
    cost_history: list[dict[str, Any]] = []
    best_cost = float("inf")
    best_cost_state: dict[str, Any] | None = None
    best_cost_epoch: int = 0
    no_improve_cost = 0
    cost_eval_every = max(0, int(train.cost_eval_every))
    cost_patience = max(0, int(train.cost_patience_evals))

    if X_val is not None and y_val is not None:
        Xv = np.asarray(X_val, dtype=np.float32)
        yv = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
    else:
        Xv = None
        yv = None
    for epoch in range(int(train.epochs)):
        epoch_losses: list[float] = []
        for xb, yb in _iterate_minibatches(X, y, int(train.batch_size), rng):
            xb_t = _t.from_numpy(xb).to(device)
            yb_t = _t.from_numpy(yb).to(device)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())
            epoch_losses.append(last_loss)
        if epoch_losses:
            loss_history.append(float(np.mean(np.asarray(epoch_losses, dtype=float))))
        else:
            loss_history.append(float("nan"))

        if Xv is not None and yv is not None:
            model.eval()
            with _t.no_grad():
                pred_v = model(_t.from_numpy(Xv).to(device))
                val_loss = float(loss_fn(pred_v, _t.from_numpy(yv).to(device)).detach().cpu().item())
            val_history.append(val_loss)

            if val_loss + min_delta < best_val:
                best_val = val_loss
                best_val_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                break

            model.train()

        if cost_eval_fn is not None and cost_eval_every > 0:
            if (epoch + 1) % cost_eval_every == 0:
                model.eval()
                try:
                    cost_out = dict(cost_eval_fn(model))
                except Exception as e:  # pragma: no cover
                    cost_out = {"error": str(e)}
                model.train()

                cost_val = float(cost_out.get("cost_total", float("inf")))
                improved = bool(np.isfinite(cost_val) and (cost_val + 1e-12 < best_cost))
                if improved:
                    best_cost = cost_val
                    best_cost_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_cost_epoch = epoch + 1
                    no_improve_cost = 0
                else:
                    no_improve_cost += 1

                cost_out.update(
                    {
                        "epoch": int(epoch + 1),
                        "cost_total": float(cost_val),
                        "best_cost": float(best_cost),
                        "improved": bool(improved),
                        "no_improve_evals": int(no_improve_cost),
                        "val_loss_mse": float(val_history[-1]) if val_history else float("nan"),
                    }
                )
                cost_history.append(cost_out)

                if cost_patience > 0 and no_improve_cost >= cost_patience:
                    break

    restore_mode = str(train.restore_best_checkpoint_by).strip().lower()
    restore_state = best_val_state
    _restored_by = "val_mse"
    _best_epoch = best_val_epoch
    if restore_mode in {"bill_cost", "cost"} and best_cost_state is not None:
        restore_state = best_cost_state
        _restored_by = "bill_cost"
        _best_epoch = best_cost_epoch
    elif restore_mode == "auto" and best_cost_state is not None:
        restore_state = best_cost_state
        _restored_by = "bill_cost"
        _best_epoch = best_cost_epoch
    if restore_state is not None:
        model.load_state_dict(restore_state)

    _total_epochs = len(loss_history)

    return FitResult(
        model=model.eval(),
        train_loss=float(last_loss),
        loss_history=loss_history,
        val_loss_history=val_history if val_history else None,
        cost_eval_history=cost_history if cost_history else None,
        best_epoch=int(_best_epoch) if _best_epoch else None,
        restored_by=_restored_by if restore_state is not None else None,
        best_val_loss=float(best_val) if best_val < float("inf") else None,
        best_cost=float(best_cost) if best_cost < float("inf") else None,
        total_epochs=int(_total_epochs),
        best_val_mse_state=best_val_state,
        best_val_mse_epoch=int(best_val_epoch) if best_val_epoch else None,
    )


def dagger_collect_labels_with_trace(
    *,
    model: Any,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    time_features: np.ndarray,
    battery: core.BatteryParams,
    allow_grid_export: bool,
    teacher: core.TeacherConfig,
    demand_charge_rate_kw: float,
    initial_E_kwh: float,
    initial_max_peak_kw: float,
    feature_cfg: FeatureConfig,
    scales: FeatureScales,
    cfg: DAggerConfig,
    timestamps_utc: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, DAggerRoundStats, dict[str, np.ndarray]]:
    """DAgger collection with policy mixing (beta) plus a reconstructable trace.

    The trace is intended for extensive auditing and alternative plotting.
    It logs the student rollout (state/action/grid) and teacher labels at the configured stride.

    Policy mixing: at each step, with probability cfg.beta the teacher's action is
    executed (and the teacher label is obtained for free). With probability (1-beta)
    the student's action is executed. Labels are still collected at the configured
    stride for student-acted steps.
    """

    _t = _require_torch()
    load_kw = np.asarray(load_kw, dtype=float)
    tf = np.asarray(time_features, dtype=np.float32)
    T = int(load_kw.shape[0])
    if tf.shape[0] != T:
        raise ValueError("time_features length mismatch")

    rollout_T = min(int(cfg.rollout_steps), T)
    label_stride = max(1, int(cfg.label_stride))
    mpc_h = max(1, int(cfg.mpc_horizon_steps))
    beta = float(cfg.beta)
    mix_rng = np.random.default_rng(42)

    env = core.EnvConfig(allow_grid_export=bool(allow_grid_export))
    state = core.EnvState(E_kwh=float(initial_E_kwh), current_max_peak_kw=float(initial_max_peak_kw))

    X_new: list[np.ndarray] = []
    y_new: list[float] = []

    # Trace arrays
    t_idx: list[int] = []
    ts_str: list[str] = []
    load_t: list[float] = []
    is_peak_t: list[int] = []
    price_t: list[float] = []
    m2s_t: list[int] = []
    m2e_t: list[int] = []
    E_t: list[float] = []
    peak_t: list[float] = []
    a_raw_t: list[float] = []
    a_safe_t: list[float] = []
    teacher_a_t: list[float] = []
    labeled_t: list[int] = []
    grid_t: list[float] = []
    gridc_t: list[float] = []
    E_next_t: list[float] = []
    peak_next_t: list[float] = []

    def _ts(i: int) -> str:
        if timestamps_utc is None:
            return ""
        try:
            return str(timestamps_utc[i])
        except Exception:
            return ""

    model.eval()
    device = _model_device(model)
    with _t.no_grad():
        for t in range(rollout_T):
            # student action
            x_t = make_features(
                load_kw=load_kw[t : t + 1],
                state_E_kwh=np.asarray([state.E_kwh], dtype=float),
                state_current_max_peak_kw=np.asarray([state.current_max_peak_kw], dtype=float),
                compiled=core.CompiledTariff(
                    is_peak_window=np.asarray([compiled.is_peak_window[t]], dtype=int),
                    energy_price=np.asarray([compiled.energy_price[t]], dtype=float),
                    minutes_to_window_start=np.asarray([compiled.minutes_to_window_start[t]], dtype=int),
                    minutes_to_window_end=np.asarray([compiled.minutes_to_window_end[t]], dtype=int),
                ),
                time_features=tf[t : t + 1],
                cfg=feature_cfg,
                battery_E_max_kwh=float(battery.E_max_kwh),
                scales=scales,
            )
            pred = model(_t.from_numpy(x_t).to(device))
            action = float(pred.detach().cpu().numpy().reshape(-1)[0])

            # Policy mixing: decide who acts this step
            use_teacher = bool(beta >= 1.0 or (beta > 0.0 and mix_rng.random() < beta))

            # We need the teacher action for labeling OR for acting
            need_teacher = use_teacher or ((t % label_stride) == 0)
            ta = float("nan")
            if need_teacher:
                end = min(T, t + mpc_h)
                mpc = core.solve_day_ahead_mpc(
                    load_kw=load_kw[t:end],
                    energy_price_per_kwh=compiled.energy_price[t:end],
                    is_peak_window=compiled.is_peak_window[t:end],
                    demand_charge_rate_kw=float(demand_charge_rate_kw),
                    current_E_kwh=float(state.E_kwh),
                    current_max_peak_kw=float(state.current_max_peak_kw),
                    battery=battery,
                    allow_grid_export=bool(allow_grid_export),
                    teacher=teacher,
                )
                ta = float(mpc.P_batt_kw[0])

            # Collect label at stride (or whenever teacher was queried)
            was_labeled = int(need_teacher)
            if need_teacher:
                X_new.append(x_t.reshape(1, -1)[0])
                y_new.append(float(ta))

            # Execute: teacher action if mixing chose teacher, else student action
            executed_action = ta if use_teacher else action

            step = core.step_env(
                state=state,
                load_kw=float(load_kw[t]),
                P_batt_kw_raw=float(executed_action),
                is_peak_window=int(compiled.is_peak_window[t]),
                battery=battery,
                env=env,
            )

            t_idx.append(int(t))
            ts_str.append(_ts(t))
            load_t.append(float(load_kw[t]))
            is_peak_t.append(int(compiled.is_peak_window[t]))
            price_t.append(float(compiled.energy_price[t]))
            m2s_t.append(int(compiled.minutes_to_window_start[t]))
            m2e_t.append(int(compiled.minutes_to_window_end[t]))
            E_t.append(float(state.E_kwh))
            peak_t.append(float(state.current_max_peak_kw))
            a_raw_t.append(float(action))
            a_safe_t.append(float(step.P_batt_kw_safe))
            teacher_a_t.append(float(ta))
            labeled_t.append(int(was_labeled))
            grid_t.append(float(step.grid_kw))
            gridc_t.append(float(step.grid_kw_clamped))
            E_next_t.append(float(step.next_state.E_kwh))
            peak_next_t.append(float(step.next_state.current_max_peak_kw))

            state = step.next_state

    if X_new:
        X_out = np.stack(X_new, axis=0).astype(np.float32)
        y_out = np.asarray(y_new, dtype=np.float32)
    else:
        X_out = np.zeros((0, 0), dtype=np.float32)
        y_out = np.zeros((0,), dtype=np.float32)

    trace = {
        "t": np.asarray(t_idx, dtype=int),
        "timestamp": np.asarray(ts_str, dtype=object),
        "load_kw": np.asarray(load_t, dtype=float),
        "is_peak_window": np.asarray(is_peak_t, dtype=int),
        "energy_price": np.asarray(price_t, dtype=float),
        "minutes_to_window_start": np.asarray(m2s_t, dtype=int),
        "minutes_to_window_end": np.asarray(m2e_t, dtype=int),
        "state_E_kwh": np.asarray(E_t, dtype=float),
        "state_current_max_peak_kw": np.asarray(peak_t, dtype=float),
        "student_action_raw_kw": np.asarray(a_raw_t, dtype=float),
        "student_action_safe_kw": np.asarray(a_safe_t, dtype=float),
        "teacher_action_kw": np.asarray(teacher_a_t, dtype=float),
        "is_labeled": np.asarray(labeled_t, dtype=int),
        "grid_kw": np.asarray(grid_t, dtype=float),
        "grid_kw_clamped": np.asarray(gridc_t, dtype=float),
        "next_E_kwh": np.asarray(E_next_t, dtype=float),
        "next_current_max_peak_kw": np.asarray(peak_next_t, dtype=float),
    }

    return X_out, y_out, DAggerRoundStats(n_labeled=int(y_out.shape[0])), trace


@dataclass(frozen=True)
class RolloutResult:
    P_batt_kw_raw: np.ndarray
    P_batt_kw_safe: np.ndarray
    E_kwh: np.ndarray
    current_max_peak_kw: np.ndarray
    grid_kw: np.ndarray
    grid_kw_clamped: np.ndarray


def rollout_policy(
    *,
    model: Any,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    time_features: np.ndarray,
    battery: core.BatteryParams,
    allow_grid_export: bool,
    initial_E_kwh: float,
    initial_max_peak_kw: float,
    feature_cfg: FeatureConfig,
    scales: FeatureScales,
    action_clip_kw: float | None = None,
) -> RolloutResult:
    _t = _require_torch()
    load_kw = np.asarray(load_kw, dtype=float)

    tf = np.asarray(time_features, dtype=np.float32)

    T = int(load_kw.shape[0])
    if tf.shape[0] != T:
        raise ValueError("time_features length mismatch")

    env = core.EnvConfig(allow_grid_export=bool(allow_grid_export))
    state = core.EnvState(E_kwh=float(initial_E_kwh), current_max_peak_kw=float(initial_max_peak_kw))

    P_raw = np.zeros(T, dtype=float)
    P_safe = np.zeros(T, dtype=float)
    E = np.zeros(T + 1, dtype=float)
    peak = np.zeros(T + 1, dtype=float)
    grid = np.zeros(T, dtype=float)
    gridc = np.zeros(T, dtype=float)
    E[0] = float(state.E_kwh)
    peak[0] = float(state.current_max_peak_kw)

    model.eval()
    device = _model_device(model)
    with _t.no_grad():
        for t in range(T):
            x_t = make_features(
                load_kw=load_kw[t : t + 1],
                state_E_kwh=np.asarray([state.E_kwh], dtype=float),
                state_current_max_peak_kw=np.asarray([state.current_max_peak_kw], dtype=float),
                compiled=core.CompiledTariff(
                    is_peak_window=np.asarray([compiled.is_peak_window[t]], dtype=int),
                    energy_price=np.asarray([compiled.energy_price[t]], dtype=float),
                    minutes_to_window_start=np.asarray([compiled.minutes_to_window_start[t]], dtype=int),
                    minutes_to_window_end=np.asarray([compiled.minutes_to_window_end[t]], dtype=int),
                ),
                time_features=tf[t : t + 1],
                cfg=feature_cfg,
                battery_E_max_kwh=float(battery.E_max_kwh),
                scales=scales,
            )
            pred = model(_t.from_numpy(x_t).to(device))
            action = float(pred.detach().cpu().numpy().reshape(-1)[0])
            if action_clip_kw is not None:
                action = float(np.clip(action, -float(action_clip_kw), float(action_clip_kw)))
            P_raw[t] = action

            step = core.step_env(
                state=state,
                load_kw=float(load_kw[t]),
                P_batt_kw_raw=float(action),
                is_peak_window=int(compiled.is_peak_window[t]),
                battery=battery,
                env=env,
            )
            P_safe[t] = float(step.P_batt_kw_safe)
            grid[t] = float(step.grid_kw)
            gridc[t] = float(step.grid_kw_clamped)

            state = step.next_state
            E[t + 1] = float(state.E_kwh)
            peak[t + 1] = float(state.current_max_peak_kw)

    return RolloutResult(
        P_batt_kw_raw=P_raw,
        P_batt_kw_safe=P_safe,
        E_kwh=E,
        current_max_peak_kw=peak,
        grid_kw=grid,
        grid_kw_clamped=gridc,
    )


@dataclass(frozen=True)
class RuleBaselineConfig:
    theta_kw: float
    precharge_window_minutes: int


def compute_threshold_from_train_months(
    *,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    timestamps_utc: Any,
    timezone: str,
    train_months: list[int],
) -> float:
    load = np.asarray(load_kw, dtype=float)
    is_peak = np.asarray(compiled.is_peak_window, dtype=int).astype(bool)
    if load.shape != is_peak.shape:
        raise ValueError("load_kw and compiled tariff must have the same shape")
    if len(load) == 0:
        return float("inf")

    ts = np.asarray(core.pd.DatetimeIndex(timestamps_utc).tz_convert(core.ensure_zoneinfo(timezone)))
    months = core.pd.DatetimeIndex(timestamps_utc).tz_convert(core.ensure_zoneinfo(timezone)).month.to_numpy(dtype=int)
    dates = core.pd.DatetimeIndex(timestamps_utc).tz_convert(core.ensure_zoneinfo(timezone)).strftime("%Y-%m-%d").to_numpy(dtype=object)
    train_mask = np.isin(months, np.asarray(train_months, dtype=int)) if train_months else np.ones_like(months, dtype=bool)
    daily_maxima: list[float] = []
    for day in np.unique(dates[train_mask]):
        day_mask = (dates == day) & train_mask & is_peak
        if np.any(day_mask):
            daily_maxima.append(float(np.max(load[day_mask])))
    if not daily_maxima:
        return float("inf")
    return float(np.median(np.asarray(daily_maxima, dtype=float)))


def compute_rule_baseline_config(
    *,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    timestamps_utc: Any,
    timezone: str,
    train_months: list[int],
) -> RuleBaselineConfig:
    theta_kw = compute_threshold_from_train_months(
        load_kw=load_kw,
        compiled=compiled,
        timestamps_utc=timestamps_utc,
        timezone=timezone,
        train_months=train_months,
    )
    ts_local = core.pd.DatetimeIndex(timestamps_utc).tz_convert(core.ensure_zoneinfo(timezone))
    months = ts_local.month.to_numpy(dtype=int)
    dates = ts_local.strftime("%Y-%m-%d").to_numpy(dtype=object)
    train_mask = np.isin(months, np.asarray(train_months, dtype=int)) if train_months else np.ones_like(months, dtype=bool)
    peak = np.asarray(compiled.is_peak_window, dtype=int).astype(bool)
    durations: list[int] = []
    for day in np.unique(dates[train_mask]):
        day_mask = (dates == day) & train_mask
        if not np.any(day_mask):
            continue
        durations.append(int(np.sum(peak[day_mask]) * 15))
    precharge_window_minutes = max(durations) if durations else 0
    return RuleBaselineConfig(theta_kw=float(theta_kw), precharge_window_minutes=int(precharge_window_minutes))


def rule_based_tariff_aware_action(
    *,
    load_kw: float,
    state: core.EnvState,
    compiled_t: core.CompiledTariff,
    battery: core.BatteryParams,
    cfg: RuleBaselineConfig,
) -> float:
    theta = float(cfg.theta_kw)
    if int(compiled_t.is_peak_window[0]) == 0:
        if (
            int(compiled_t.minutes_to_window_start[0]) <= int(cfg.precharge_window_minutes)
            and float(state.E_kwh) < float(battery.E_max_kwh)
        ):
            return float(
                min(
                    float(battery.P_max_kw),
                    (float(battery.E_max_kwh) - float(state.E_kwh)) / (float(battery.eta_charge) * float(battery.dt_hours)),
                )
            )
        return 0.0

    if not np.isfinite(theta) or float(load_kw) <= theta:
        return 0.0

    need = float(load_kw) - theta
    discharge_max = min(
        float(battery.P_max_kw),
        float(state.E_kwh) * float(battery.eta_discharge) / float(battery.dt_hours),
    )
    return -float(min(need, discharge_max))


def rollout_rule_policy(
    *,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    battery: core.BatteryParams,
    allow_grid_export: bool,
    initial_E_kwh: float,
    initial_max_peak_kw: float,
    cfg: RuleBaselineConfig,
) -> RolloutResult:
    load = np.asarray(load_kw, dtype=float)
    T = int(load.shape[0])
    env = core.EnvConfig(allow_grid_export=bool(allow_grid_export))
    state = core.EnvState(E_kwh=float(initial_E_kwh), current_max_peak_kw=float(initial_max_peak_kw))

    P_raw = np.zeros(T, dtype=float)
    P_safe = np.zeros(T, dtype=float)
    E = np.zeros(T + 1, dtype=float)
    peak = np.zeros(T + 1, dtype=float)
    grid = np.zeros(T, dtype=float)
    gridc = np.zeros(T, dtype=float)
    E[0] = float(state.E_kwh)
    peak[0] = float(state.current_max_peak_kw)

    for t in range(T):
        compiled_t = core.CompiledTariff(
            is_peak_window=np.asarray([compiled.is_peak_window[t]], dtype=int),
            energy_price=np.asarray([compiled.energy_price[t]], dtype=float),
            minutes_to_window_start=np.asarray([compiled.minutes_to_window_start[t]], dtype=int),
            minutes_to_window_end=np.asarray([compiled.minutes_to_window_end[t]], dtype=int),
            active_schedule_id=None if compiled.active_schedule_id is None else np.asarray([compiled.active_schedule_id[t]], dtype=int),
            is_super_offpeak=None if compiled.is_super_offpeak is None else np.asarray([compiled.is_super_offpeak[t]], dtype=int),
            is_midpeak=None if compiled.is_midpeak is None else np.asarray([compiled.is_midpeak[t]], dtype=int),
        )
        action = rule_based_tariff_aware_action(
            load_kw=float(load[t]),
            state=state,
            compiled_t=compiled_t,
            battery=battery,
            cfg=cfg,
        )
        P_raw[t] = float(action)
        step = core.step_env(
            state=state,
            load_kw=float(load[t]),
            P_batt_kw_raw=float(action),
            is_peak_window=int(compiled.is_peak_window[t]),
            battery=battery,
            env=env,
        )
        P_safe[t] = float(step.P_batt_kw_safe)
        grid[t] = float(step.grid_kw)
        gridc[t] = float(step.grid_kw_clamped)
        state = step.next_state
        E[t + 1] = float(state.E_kwh)
        peak[t + 1] = float(state.current_max_peak_kw)

    return RolloutResult(
        P_batt_kw_raw=P_raw,
        P_batt_kw_safe=P_safe,
        E_kwh=E,
        current_max_peak_kw=peak,
        grid_kw=grid,
        grid_kw_clamped=gridc,
    )


@dataclass(frozen=True)
class DAggerConfig:
    enabled: bool = False
    rollout_steps: int = 2000
    label_stride: int = 1
    mpc_horizon_steps: int = 96
    beta: float = 0.0  # policy mixing: P(teacher acts) per step; 0=pure student, 1=pure teacher


@dataclass(frozen=True)
class RuntimeStats:
    mean_ms: float
    p95_ms: float


def measure_policy_inference_ms(*, model: Any, X: np.ndarray, repeats: int = 100) -> RuntimeStats:
    _t = _require_torch()
    X = np.asarray(X, dtype=np.float32)
    model.eval()
    device = _model_device(model)

    times: list[float] = []
    with _t.no_grad():
        for _ in range(int(repeats)):
            t0 = perf_counter()
            _ = model(_t.from_numpy(X).to(device))
            t1 = perf_counter()
            times.append((t1 - t0) * 1000.0)

    arr = np.asarray(times, dtype=float)
    return RuntimeStats(mean_ms=float(np.mean(arr)), p95_ms=float(np.percentile(arr, 95)))


@dataclass(frozen=True)
class DAggerRoundStats:
    n_labeled: int
    teacher_bill_total: float | None = None


def dagger_collect_labels(
    *,
    model: Any,
    load_kw: np.ndarray,
    compiled: core.CompiledTariff,
    time_features: np.ndarray,
    battery: core.BatteryParams,
    allow_grid_export: bool,
    teacher: core.TeacherConfig,
    demand_charge_rate_kw: float,
    initial_E_kwh: float,
    initial_max_peak_kw: float,
    feature_cfg: FeatureConfig,
    scales: FeatureScales,
    cfg: DAggerConfig,
) -> tuple[np.ndarray, np.ndarray, DAggerRoundStats]:
    """Roll out with policy mixing (beta), collect (state, teacher_action) pairs."""

    _t = _require_torch()
    load_kw = np.asarray(load_kw, dtype=float)
    tf = np.asarray(time_features, dtype=np.float32)
    T = int(load_kw.shape[0])
    if tf.shape[0] != T:
        raise ValueError("time_features length mismatch")

    rollout_T = min(int(cfg.rollout_steps), T)
    label_stride = max(1, int(cfg.label_stride))
    mpc_h = max(1, int(cfg.mpc_horizon_steps))
    beta = float(cfg.beta)
    mix_rng = np.random.default_rng(42)

    env = core.EnvConfig(allow_grid_export=bool(allow_grid_export))
    state = core.EnvState(E_kwh=float(initial_E_kwh), current_max_peak_kw=float(initial_max_peak_kw))

    X_new: list[np.ndarray] = []
    y_new: list[float] = []

    model.eval()
    device = _model_device(model)
    with _t.no_grad():
        for t in range(rollout_T):
            # student action
            x_t = make_features(
                load_kw=load_kw[t : t + 1],
                state_E_kwh=np.asarray([state.E_kwh], dtype=float),
                state_current_max_peak_kw=np.asarray([state.current_max_peak_kw], dtype=float),
                compiled=core.CompiledTariff(
                    is_peak_window=np.asarray([compiled.is_peak_window[t]], dtype=int),
                    energy_price=np.asarray([compiled.energy_price[t]], dtype=float),
                    minutes_to_window_start=np.asarray([compiled.minutes_to_window_start[t]], dtype=int),
                    minutes_to_window_end=np.asarray([compiled.minutes_to_window_end[t]], dtype=int),
                ),
                time_features=tf[t : t + 1],
                cfg=feature_cfg,
                battery_E_max_kwh=float(battery.E_max_kwh),
                scales=scales,
            )
            pred = model(_t.from_numpy(x_t).to(device))
            action = float(pred.detach().cpu().numpy().reshape(-1)[0])

            # Policy mixing: decide who acts this step
            use_teacher = bool(beta >= 1.0 or (beta > 0.0 and mix_rng.random() < beta))

            # Need teacher for labeling at stride or if teacher is acting
            need_teacher = use_teacher or ((t % label_stride) == 0)
            if need_teacher:
                end = min(T, t + mpc_h)
                mpc = core.solve_day_ahead_mpc(
                    load_kw=load_kw[t:end],
                    energy_price_per_kwh=compiled.energy_price[t:end],
                    is_peak_window=compiled.is_peak_window[t:end],
                    demand_charge_rate_kw=float(demand_charge_rate_kw),
                    current_E_kwh=float(state.E_kwh),
                    current_max_peak_kw=float(state.current_max_peak_kw),
                    battery=battery,
                    allow_grid_export=bool(allow_grid_export),
                    teacher=teacher,
                )
                X_new.append(x_t.reshape(1, -1)[0])
                y_new.append(float(mpc.P_batt_kw[0]))

            executed_action = float(mpc.P_batt_kw[0]) if (use_teacher and need_teacher) else action

            step = core.step_env(
                state=state,
                load_kw=float(load_kw[t]),
                P_batt_kw_raw=float(executed_action),
                is_peak_window=int(compiled.is_peak_window[t]),
                battery=battery,
                env=env,
            )
            state = step.next_state

    if X_new:
        X_out = np.stack(X_new, axis=0).astype(np.float32)
        y_out = np.asarray(y_new, dtype=np.float32)
    else:
        X_out = np.zeros((0, 0), dtype=np.float32)
        y_out = np.zeros((0,), dtype=np.float32)

    return X_out, y_out, DAggerRoundStats(n_labeled=int(y_out.shape[0]))
