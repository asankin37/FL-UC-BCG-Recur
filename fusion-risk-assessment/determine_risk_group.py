"""Determine the uncovered clusters as the low-risk and high-risk groups."""


import argparse
import os
import json
import pandas as pd
from lifelines import CoxPHFitter


def main(
	metadatapath: str, survivalpath: str, clusterpath: str, savepath: str, by_cox_regression: bool
) -> None:
	"""Save a mapping of the cluster ID onto low-risk or high-risk group."""
	metadata = pd.read_csv(
		metadatapath
	).merge(
		pd.read_csv(survivalpath), how="inner", on="case_id"
	).merge(
		pd.read_csv(clusterpath), how="inner", on="slide_id"
	)
	if by_cox_regression:
		cph = CoxPHFitter()
		cph.fit(
			metadata[["cluster_id", "relapse_free_time", "relapsed"]],
			duration_col="relapse_free_time",
			event_col="relapsed"
		)
		# Reverse ordering if the hazard ratio is less than one
		risk_ordering = sorted(
			metadata["cluster_id"].unique(),
			reverse=cph.summary.loc["cluster_id", "exp(coef)"] < 1
		)
	else:
		risk_ordering = metadata.groupby(
			"cluster_id"
		)["relapse_free_time"].mean().sort_values(ascending=False).index
	mapping = {
		str(risk_ordering[0]): "low-risk",
		str(risk_ordering[1]): "high-risk"
	} if len(risk_ordering) >= 2 else {
		str(risk_ordering[0]): "low-risk"  # default to low-risk
	}
	with open(savepath, "w") as fp:
		json.dump(mapping, fp)


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
		"--clusterpath",
		type=str,
		required=True,
		help="File path to clustering assignment of slides"
	)
	parser.add_argument(
		"--savepath",
		type=str,
		required=True,
		help="File path to save mapping between clusters and risk groups"
	)
	parser.add_argument(
		"--by_cox_regression",
		default=False,
		action="store_true",
		help="Whether to determine low/high risk by Cox regression"
	)
	args = parser.parse_args()
	os.makedirs(os.path.dirname(args.savepath), exist_ok=True)
	main(
		metadatapath=args.metadatapath,
		survivalpath=args.survivalpath,
		clusterpath=args.clusterpath,
		savepath=args.savepath,
		by_cox_regression=args.by_cox_regression
	)

