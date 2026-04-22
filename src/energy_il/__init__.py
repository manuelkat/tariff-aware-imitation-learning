"""Energy IL package (tariff-aware imitation learning).

Consolidated to a small number of modules:
- `energy_il.core`: tariffs, billing, env, teacher, data utils
- `energy_il.student`: features, BC/DAgger-lite, rollouts
- `energy_il.plots`: plotting helpers

The package root re-exports the most commonly used symbols.
"""

from .core import (  # noqa: F401
	BatteryParams,
	BillBreakdown,
	CalendarMask,
	CompiledTariff,
	DemandWindowRule,
	EnvConfig,
	EnvState,
	EnergyPriceRule,
	ForecastNoiseConfig,
	LoadedTimeseries,
	MPCResult,
	MonthSplit,
	PeakWindow,
	SeasonalTariffBlock,
	SolverLog,
	StepResult,
	TariffIR,
	TeacherConfig,
	TeacherRollout,
	apply_ar1_forecast_noise,
	basic_time_features,
	clip_action_to_soc_bounds,
	compile_tariff,
	compute_monthly_bill,
	load_processed_timeseries_csv,
	make_experiment_tariff_suite,
	make_paper_suite,
	make_tariff,
	run_teacher_receding_horizon,
	solve_day_ahead_mpc,
	split_by_month,
	split_by_month_lists,
	step_env,
	tariff_family_from_name,
)

from .student import (  # noqa: F401
	DAggerConfig,
	DAggerRoundStats,
	FeatureConfig,
	FeatureScales,
	FitResult,
	MLPConfig,
	MLPPolicy,
	RuleBaselineConfig,
	RolloutResult,
	RuntimeStats,
	TrainConfig,
	compute_rule_baseline_config,
	compute_threshold_from_train_months,
	dagger_collect_labels,
	dagger_collect_labels_with_trace,
	fit_bc_policy,
	make_features,
	measure_policy_inference_ms,
	rollout_rule_policy,
	rollout_policy,
	rule_based_tariff_aware_action,
)
