"""Perform log-rank test of relapse-free survival data among different groupings."""


import argparse
import os
from collections import namedtuple
import pandas as pd
from lifelines.statistics import logrank_test
from _analysis_utils import read_survival_data_grouping


def logrank_test_result(
	data: pd.DataFrame, grouping: str
) -> tuple[float, float] | tuple[None, None]:
	"""Return test statistic and p-value of log-rank test."""
	groups = data[grouping].drop_duplicates().sort_values()
	if len(groups) < 2:
		return None, None
	grp_1, grp_2, *_ = groups
	result = logrank_test(
		durations_A=data.loc[data[grouping] == grp_1, "relapse_free_time"],
		durations_B=data.loc[data[grouping] == grp_2, "relapse_free_time"],
		event_observed_A=data.loc[data[grouping] == grp_1, "relapsed"],
		event_observed_B=data.loc[data[grouping] == grp_2, "relapsed"],
	)
	return result.test_statistic, result.p_value


TestResult = namedtuple("TestResult", ["Feature", "TestStatistic", "PValue"])


def main(
	metadatapath: str,
	survivalpath: str,
	clusterdir: str,
	riskgroupdir: str | None,
	num_folds: int,
	savepath: str,
) -> None:
	"""
	Save log-rank test statistics and p-value for grouping of (a) tumor stage,
	(b) tumor grade, and (c) clustering based on slide features predicted from
	CLAM models.
	"""
	data = read_survival_data_grouping(
		metadatapath, survivalpath, clusterdir, riskgroupdir, num_folds
	)
	# Perform log-rank test for grouping of tumor stage, tumor grande, and
	# clustering by slide embedding of each fold
	results = pd.DataFrame(
		[
			TestResult("Tumor stage", *logrank_test_result(data, "uc_stage")),
			TestResult("Tumor grade", *logrank_test_result(data, "uc_grade")),
			*[
				(
					TestResult(
						f"Fusion risk factor (fold {fold_id})",
						*logrank_test_result(data, f"fold_{fold_id}_cluster_id")
					)
					if num_folds > 1 else
					TestResult(
						"Fusion risk factor",
						*logrank_test_result(data, "cluster_id")
					)
				)
				for fold_id in range(num_folds)
			]
		]
	).round(decimals=4)
	results = results.rename(
		columns={"TestStatistic": "Test Statistic", "PValue": "p-Value"}
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
		default=1,
		help="Number of folds of CLAM models that predicted slide embedding"
	)
	parser.add_argument(
		"--savepath",
		type=str,
		required=True,
		help="File path to save table of test results"
	)
	args = parser.parse_args()
	os.makedirs(os.path.dirname(args.savepath), exist_ok=True)
	main(
		metadatapath=args.metadatapath,
		survivalpath=args.survivalpath,
		clusterdir=args.clusterdir,
		riskgroupdir=args.riskgroupdir,
		num_folds=args.num_folds,
		savepath=args.savepath
	)

