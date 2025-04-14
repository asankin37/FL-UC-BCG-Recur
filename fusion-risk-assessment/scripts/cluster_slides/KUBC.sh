#!/bin/bash

set -e

# Cluster CLAM cohort
python cluster_slides.py \
	--slide_featdir_fit "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--slide_featdir_pred "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--linear_transform_dir "../artifact/slide_embedding/FL_foldless_linear_transform" \
	--seed 12345 \
	--savepath "../artifact/clustering/clam_cohort/FL_foldless_on_KUBC/fold_0.csv" \
	--savepath_probden "../artifact/clustering/clam_cohort/FL_foldless_on_KUBC/fold_0_prob.csv"
# Cluster survival analysis cohort
python cluster_slides.py \
	--slide_featdir_fit "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--slide_featdir_pred "../artifact/slide_embedding/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--linear_transform_dir "../artifact/slide_embedding/FL_foldless_linear_transform" \
	--seed 12345 \
	--savepath "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC/fold_0.csv" \
	--savepath_probden "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC/fold_0_prob.csv"
# Determine low-risk and high-risk cluster
python determine_risk_group.py \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--survivalpath "../external/relapse_free_survival.csv" \
	--clusterpath "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC/fold_0.csv" \
	--savepath "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC/fold_0.json"
# Visualize clustering results of slide embedding
python visualize_slide_embedding.py \
	--slide_featdir_fit "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--slide_featdir_pred "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--metadatapath "../artifact/metadata_clam_cohort.csv" \
	--clusterpath "../artifact/clustering/clam_cohort/FL_foldless_on_KUBC/fold_0.csv" \
	--riskgrouppath "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC/fold_0.json" \
	--linear_transform_dir "../artifact/slide_embedding/FL_foldless_linear_transform" \
	--seed 12345 \
	--savepath "../analysis/clustering/FL_foldless_on_KUBC/clam_cohort.png"
python visualize_slide_embedding.py \
	--slide_featdir_fit "../artifact/slide_embedding/clam_cohort/FL_foldless_on_KUBC" \
	--slide_featdir_pred "../artifact/slide_embedding/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--clusterpath "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC/fold_0.csv" \
	--riskgrouppath "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC/fold_0.json" \
	--linear_transform_dir "../artifact/slide_embedding/FL_foldless_linear_transform" \
	--seed 12345 \
	--savepath "../analysis/clustering/FL_foldless_on_KUBC/survival_analysis_cohort.png"
