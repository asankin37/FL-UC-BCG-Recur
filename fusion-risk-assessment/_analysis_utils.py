"""Utilities for survival analysis."""


import os
import json
import pandas as pd


pd.set_option("future.no_silent_downcasting", True)


def _clusterpath_of_fold(dirpath: str, fold_id: int) -> str:
	return os.path.join(dirpath, f"fold_{fold_id}.csv")


def _riskgrppath_of_fold(dirpath: str, fold_id: int) -> str:
	return os.path.join(dirpath, f"fold_{fold_id}.json")


def _riskgroup_clustering(clstrpath: str, riskgrppath: str) -> pd.DataFrame:
	with open(riskgrppath, "r") as fp:
		mapping = json.load(fp)
	mapping = {
		int(ind): 0 if grp == "low-risk" else 1
		for ind, grp in mapping.items()
	}
	return pd.read_csv(clstrpath).replace({"cluster_id": mapping})


def _clstr_ordered_by_rfs(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
	_data = data.copy()
	mapping = {}
	for col in cols:
		ordering = _data.groupby(col)["relapse_free_time"].mean().sort_values(ascending=False).index
		mapping[col] = {clstr: order for order, clstr in enumerate(ordering)}
	return _data.replace(mapping)


def read_survival_data_grouping(
	metadatapath: str,
	survivalpath: str,
	clusterdir: str,
	riskgroupdir: str | None,
	num_folds: int
) -> pd.DataFrame:
	"""
	Return survival metadata along with the predicted clustering of slides.

	Arguments
	---------
	metadatapath: File path to the metadata of slides
	survivalpath: File path to the survival data of slides
	clusterdir: Directory containing clustering assignment of slides in each fold
	riskgroupdir: Directory containing mapping between risk groups and clusters
	num_folds: Number of folds of CLAM models that predicted slide embedding
	"""
	# Read metadata of slides
	metadata = pd.merge(
		pd.read_csv(metadatapath),
		pd.read_csv(survivalpath),
		on="case_id",
		how="inner"
	)[["case_id", "slide_id", "uc_stage", "uc_grade", "relapse_free_time", "relapsed"]]
	metadata["relapse_free_time"] /= 365  # days to years
	metadata = metadata.replace(  # binarize class labels
		{
			"uc_stage": {"non-invasive": 0, "infiltrating": 1},
			"uc_grade": {"low-grade": 0, "high-grade": 1}
		}
	)
	# Read clustering result of slides
	clustering = metadata["slide_id"].to_frame()
	for fold_id in range(num_folds):
		assignment = (
			_riskgroup_clustering(
				clstrpath=_clusterpath_of_fold(clusterdir, fold_id),
				riskgrppath=_riskgrppath_of_fold(riskgroupdir, fold_id)
			)
			if riskgroupdir is not None else
			pd.read_csv(_clusterpath_of_fold(clusterdir, fold_id))
		)
		if num_folds > 1:
			assignment = assignment.rename(
				columns={"cluster_id": f"fold_{fold_id}_cluster_id"}
			)
		clustering = clustering.merge(assignment, how="inner", on="slide_id")
	# Merge metadata and clustering
	data = metadata.merge(clustering, on="slide_id", how="inner")
	if riskgroupdir is None:  # clusters have not been mapped to risk groups
		if num_folds > 1:
			data = _clstr_ordered_by_rfs(
				data,
				[f"fold_{fold_id}_cluster_id" for fold_id in range(num_folds)]
			)
		else:
			data = _clstr_ordered_by_rfs(data, ["cluster_id"])
	return data


def main() -> None:
	"""Empty main."""


if __name__ == "__main__":
	main()

