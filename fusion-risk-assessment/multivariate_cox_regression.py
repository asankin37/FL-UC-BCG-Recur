"""Perform multivariate Cox regression with different ways of grouping."""


import argparse
import os
from collections import namedtuple
from collections.abc import Iterator
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from _analysis_utils import read_survival_data_grouping


FEATURES = ["uc_stage", "uc_grade"]
FEATURE_LABEL = {"uc_stage": "Tumor stage", "uc_grade": "Tumor grade"}


def _check_proportional_hazard(data: pd.DataFrame, *covariates: str) -> None:
	"""Verify proportional hazard assumption of multivariate Cox model."""
	_data = data[[*covariates, "relapse_free_time", "relapsed"]]
	cph = CoxPHFitter()
	cph.fit(
		_data,
		duration_col="relapse_free_time",
		event_col="relapsed"
	)
	print(f"Cox regression with covariates {covariates}")
	cph.check_assumptions(_data, p_value_threshold=0.05)


def cox_regr_per_covariate_result(
	cph: CoxPHFitter, *covariates: str
) -> Iterator[tuple[float, float, float, float]]:
	"""
	Return an iterator of the hazard ratio, p-value, and the 95% confidence
	interval for each covariate.
	"""
	summary = cph.summary
	for cvrt in covariates:
		yield (
			summary.loc[cvrt, "exp(coef)"],
			summary.loc[cvrt, "p"],
			summary.loc[cvrt, "exp(coef) lower 95%"],
			summary.loc[cvrt, "exp(coef) upper 95%"]
		)


def cox_regr_overall_stats(cph: CoxPHFitter) -> tuple[float, float]:
	"""Return the concordance and the partial AIC."""
	return float(cph.concordance_index_), float(cph.AIC_partial_)


def multivariate_cox_regr_result(
	data: pd.DataFrame, *covariates: str
) -> tuple[
		Iterator[tuple[float, float, float, float] | tuple[None, None, None, None]],
		tuple[float, float] | tuple[None, None]
	]:
	"""
	Return the hazard ratio, p-value, and the 95% confidence interval for each
	covariate as an iterator, and also oreturn the concordance and partial AIC
	for the model.
	"""
	try:
		_data = data[[*covariates, "relapse_free_time", "relapsed"]].dropna()
		cph = CoxPHFitter()
		cph.fit(_data, duration_col="relapse_free_time", event_col="relapsed")
		return (
			cox_regr_per_covariate_result(cph, *covariates),
			cox_regr_overall_stats(cph)
		)
	except ConvergenceError as e:
		print(
			f"Regression fitting failed with covariates {covariates}: {str(e)}"
		)
		return (
			((None, None, None, None) for _ in covariates),
			(None, None)
		)


PerCovariateResult = namedtuple(
	"PerCovariateResult",
	["Model", "Feature", "HazardRatio", "PValue", "LowerCi", "UpperCi"]
)


OverallResult = namedtuple(
	"OverallResult", ["Model", "Concordance", "AIC"]
)


def main(
	metadatapath: str,
	survivalpath: str,
	clusterdir: str,
	riskgroupdir: str | None,
	num_folds: int,
	savedir: str,
	saveprefix: str,
	check_assumption: bool = False,
	additional_metadatapath: str | None = None,
	additional_features: list[str] | None = None
) -> None:
	"""
	Save the hazard ratio, p-value, concordance, and AIC in multivariate Cox
	regression among the assigned clusters and different groupings. These
	results are saved at the `<saveprefix>_per_feature.csv` and `<saveprefix>_
	overall.csv` in `savedir` respectively.
	"""
	data = read_survival_data_grouping(
		metadatapath, survivalpath, clusterdir, riskgroupdir, num_folds
	)
	if additional_metadatapath is not None:
		data = data.merge(
			pd.read_csv(additional_metadatapath), how="inner", on="case_id"
		)
	# Check proportional hazard assumption of multivariate Cox models
	if check_assumption:
		for fold_id in range(num_folds):
			clstrcol = (
				f"fold_{fold_id}_cluster_id"
				if num_folds > 1 else
				"cluster_id"
			)
			_check_proportional_hazard(data, "uc_stage", clstrcol)
			_check_proportional_hazard(data, "uc_grade", clstrcol)
			if additional_features is not None:
				for feature in additional_features:
					_check_proportional_hazard(data, feature, clstrcol)
	# Collect hazard ratio and p-value for each covariate, along with the
	# overall concordance and AIC
	features = (
		FEATURES
		if additional_features is None else
		FEATURES + additional_features
	)
	per_feature_results, overall_results = [], []
	for fold_id in range(num_folds):
		clstrcol = (
			f"fold_{fold_id}_cluster_id"
			if num_folds > 1 else
			"cluster_id"
		)
		clstrlbl = (
			f"Fusion risk factor (fold {fold_id})"
			if num_folds > 1 else
			f"Fusion risk factor"
		)
		for feature in features:
			featlbl = FEATURE_LABEL.get(feature, feature)
			modellbl = f"{featlbl} + {clstrlbl}"
			(feat_res, clstr_res), overall_res = multivariate_cox_regr_result(
				data, feature, clstrcol
			)
			per_feature_results += [
				PerCovariateResult(modellbl, featlbl, *feat_res),
				PerCovariateResult(modellbl, clstrlbl, *clstr_res),
			]
			overall_results.append(OverallResult(modellbl, *overall_res))
	# Save results of Cox regression
	per_feature_results = pd.DataFrame(
		per_feature_results
	).round(
		decimals=4
	).rename(
		columns={
			"HazardRatio": "Hazard Ratio",
			"PValue": "p-Value",
			"LowerCi": "Lower 95% CI",
			"UpperCi": "Upper 95% CI"
		}
	)
	overall_results = pd.DataFrame(
		overall_results
	).round(
		decimals=2
	).rename(
		columns={"Concordance": "C-index"}
	)
	per_feature_results.to_csv(
		os.path.join(savedir, f"{saveprefix}_per_feature.csv"), index=False
	)
	print(per_feature_results)
	overall_results.to_csv(
		os.path.join(savedir, f"{saveprefix}_overall.csv"), index=False
	)
	print(overall_results)


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
		"--savedir",
		type=str,
		required=True,
		help="Directory to save tables of Cox regression results"
	)
	parser.add_argument(
		"--saveprefix",
		type=str,
		required=True,
		help="File name prefix of the saved talbes of Cox regression results"
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
	os.makedirs(args.savedir, exist_ok=True)
	main(
		metadatapath=args.metadatapath,
		survivalpath=args.survivalpath,
		clusterdir=args.clusterdir,
		riskgroupdir=args.riskgroupdir,
		num_folds=args.num_folds,
		savedir=args.savedir,
		saveprefix=args.saveprefix,
		check_assumption=args.check_model_assumption,
		additional_metadatapath=args.additional_metadatapath,
		additional_features=args.additional_features
	)

