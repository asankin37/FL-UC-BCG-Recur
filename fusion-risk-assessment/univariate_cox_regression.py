"""Perform univariate Cox regression with different ways of grouping."""


import argparse
import os
from collections import namedtuple
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from _analysis_utils import read_survival_data_grouping


def _check_proportional_hazard(data: pd.DataFrame, grouping: str) -> None:
	"""Verify proportional hazard assumption of univariate Cox model."""
	_data = data[[grouping, "relapse_free_time", "relapsed"]]
	cph = CoxPHFitter()
	cph.fit(
		_data,
		duration_col="relapse_free_time",
		event_col="relapsed"
	)
	print(f"Grouped by {grouping}")
	cph.check_assumptions(_data, p_value_threshold=0.05)


def univariate_cox_regr_result(
	data: pd.DataFrame, grouping: str
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
	"""
	Return hazard ratio, p-value, and 95% confidence interval of the univariate
	Cox regression.
	"""
	try:
		_data = data[[grouping, "relapse_free_time", "relapsed"]].dropna()
		cph = CoxPHFitter()
		cph.fit(_data, duration_col="relapse_free_time", event_col="relapsed")
		summary = cph.summary
		return (
			summary.loc[grouping, "exp(coef)"],
			summary.loc[grouping, "p"],
			summary.loc[grouping, "exp(coef) lower 95%"],
			summary.loc[grouping, "exp(coef) upper 95%"]
		)
	except ConvergenceError as e:
		print(f"Regression fitting failed with grouping {grouping}: {str(e)}")
		return None, None, None, None


CoxRegrResult = namedtuple(
	"CoxRegrResult", ["Feature", "HazardRatio", "PValue", "LowerCi", "UpperCi"]
)


def main(
	metadatapath: str,
	survivalpath: str,
	clusterdir: str,
	riskgroupdir: str | None,
	num_folds: int,
	savepath: str,
	check_assumption: bool = False,
	additional_metadatapath: str | None = None,
	additional_features: list[str] | None = None
) -> None:
	"""
	Save the hazard ratio and the p-value in univariate Cox regression of
	different groupings.
	"""
	data = read_survival_data_grouping(
		metadatapath, survivalpath, clusterdir, riskgroupdir, num_folds
	)
	if additional_metadatapath is not None:
		data = data.merge(
			pd.read_csv(additional_metadatapath), how="inner", on="case_id"
		)
	# Check proportional hazard assumption of univariate Cox models
	if check_assumption:
		_check_proportional_hazard(data, "uc_stage")
		_check_proportional_hazard(data, "uc_grade")
		for fold_id in range(num_folds):
			_check_proportional_hazard(
				data,
				f"fold_{fold_id}_cluster_id" if num_folds > 1 else "cluster_id"
			)
		if additional_features is not None:
			for feature in additional_features:
				_check_proportional_hazard(data, feature)
	# Collect hazard ratio and p-value
	results = [
		CoxRegrResult(
			"Tumor stage", *univariate_cox_regr_result(data, "uc_stage")
		),
		CoxRegrResult(
			"Tumor grade", *univariate_cox_regr_result(data, "uc_grade")
		),
		*[
			(
				CoxRegrResult(
					f"Fusion risk factor (fold {fold_id})",
					*univariate_cox_regr_result(
						data, f"fold_{fold_id}_cluster_id"
					)
				)
				if num_folds > 1 else
				CoxRegrResult(
					"Fusion risk factor",
					*univariate_cox_regr_result(data, "cluster_id")
				)
			)
			for fold_id in range(num_folds)
		]
	]
	if additional_features is not None:
		results += [
			CoxRegrResult(feature, *univariate_cox_regr_result(data, feature))
			for feature in additional_features
		]
	# Save results of Cox regression
	results = pd.DataFrame(results).round(decimals=4)
	results = results.rename(
		columns={
			"HazardRatio": "Hazard Ratio",
			"PValue": "p-Value",
			"LowerCi": "Lower 95% CI",
			"UpperCi": "Upper 95% CI"
		}
	)
	results.to_csv(savepath, index=False)
	print(results)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--metadatapath",
		type=str,
		required=True,
		help="File path to the metadata of slides"
	)
	parser.add_argument(
		"--survivalpath",
		type=str,
		required=True,
		help="File path to the survival data"
	)
	parser.add_argument(
		"--clusterdir",
		type=str,
		required=True,
		help="Directory containing clustering assignment of slides in each fold"
	)
	parser.add_argument(
		"--riskgroupdir",
		type=str,
		default=None,
		help="Directory containing mapping between clusters and risk groups"
	)
	parser.add_argument(
		"--num_folds",
		type=int,
		required=True,
		help="Number of folds of CLAM models that predicted slide embedding"
	)
	parser.add_argument(
		"--savepath",
		type=str,
		required=True,
		help="File path to save table of Cox regression results"
	)
	parser.add_argument(
		"--check_model_assumption",
		default=False,
		action="store_true",
		help=(
			"Whether to check proportional hazard assumption of univariate "
			"Cox regression  model"
		)
	)
	parser.add_argument(
		"--additional_metadatapath",
		type=str,
		default=None,
		help=(
			"File path to additional metadata of slides; additional metadata "
			"must contain a 'case_id' column"
		)
	)
	parser.add_argument(
		"--additional_features",
		type=str,
		nargs="+",
		default=None,
		help=(
			"Additional features each as a covariate of Cox regression; must "
			"be a column name of the additional metadata"
		)
	)
	args = parser.parse_args()
	os.makedirs(os.path.dirname(args.savepath), exist_ok=True)
	main(
		metadatapath=args.metadatapath,
		survivalpath=args.survivalpath,
		clusterdir=args.clusterdir,
		riskgroupdir=args.riskgroupdir,
		num_folds=args.num_folds,
		savepath=args.savepath,
		check_assumption=args.check_model_assumption,
		additional_metadatapath=args.additional_metadatapath,
		additional_features=args.additional_features
	)

