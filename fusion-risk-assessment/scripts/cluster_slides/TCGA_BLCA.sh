#!/bin/bash

set -e

models=(
	"FL_foldless"
)
for model in "${models[@]}"; do
	# Cluster fitting cohort
	python cluster_slides.py \
		--slide_featdir_fit "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--slide_featdir_pred "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--linear_transform_dir "../artifact/slide_embedding/${model}_linear_transform" \
		--seed 12345 \
		--savepath "../artifact/TCGA_BLCA/clustering/clam_cohort/${model}_on_TCGA_BLCA/fold_0.csv" \
		--savepath_probden "../artifact/TCGA_BLCA/clustering/clam_cohort/${model}_on_TCGA_BLCA/fold_0_prob.csv"
	# Cluster survival analysis cohort
	python cluster_slides.py \
		--slide_featdir_fit "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--slide_featdir_pred "../artifact/TCGA_BLCA/slide_embedding/survival_analysis_cohort/${model}_on_TCGA_BLCA" \
		--linear_transform_dir "../artifact/slide_embedding/${model}_linear_transform" \
		--seed 12345 \
		--savepath "../artifact/TCGA_BLCA/clustering/survival_analysis_cohort/${model}_on_TCGA_BLCA/fold_0.csv" \
		--savepath_probden "../artifact/TCGA_BLCA/clustering/survival_analysis_cohort/${model}_on_TCGA_BLCA/fold_0_prob.csv"
	# Determine low-risk and high-risk cluster
	python determine_risk_group.py \
		--metadatapath "../artifact/TCGA_BLCA/metadata_clam_cohort.csv" \
		--survivalpath "../external/TCGA_BLCA/survival.csv" \
		--clusterpath "../artifact/TCGA_BLCA/clustering/clam_cohort/${model}_on_TCGA_BLCA/fold_0.csv" \
		--savepath "../artifact/TCGA_BLCA/clustering/risk_group_mapping/${model}_on_TCGA_BLCA/fold_0.json" \
		--by_cox_regression
	# Visualize clustering results of slide embedding
	python visualize_slide_embedding.py \
		--slide_featdir_fit "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--slide_featdir_pred "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--metadatapath "../artifact/TCGA_BLCA/metadata_clam_cohort.csv" \
		--clusterpath "../artifact/TCGA_BLCA/clustering/clam_cohort/${model}_on_TCGA_BLCA/fold_0.csv" \
		--riskgrouppath "../artifact/TCGA_BLCA/clustering/risk_group_mapping/${model}_on_TCGA_BLCA/fold_0.json" \
		--linear_transform_dir "../artifact/slide_embedding/${model}_linear_transform" \
		--seed 12345 \
		--savepath "../analysis/TCGA_BLCA/clustering/${model}_on_TCGA_BLCA/clam_cohort.png"
	python visualize_slide_embedding.py \
		--slide_featdir_fit "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/${model}_on_TCGA_BLCA" \
		--slide_featdir_pred "../artifact/TCGA_BLCA/slide_embedding/survival_analysis_cohort/${model}_on_TCGA_BLCA" \
		--metadatapath "../artifact/TCGA_BLCA/metadata_survival_analysis_cohort.csv" \
		--clusterpath "../artifact/TCGA_BLCA/clustering/survival_analysis_cohort/${model}_on_TCGA_BLCA/fold_0.csv" \
		--riskgrouppath "../artifact/TCGA_BLCA/clustering/risk_group_mapping/${model}_on_TCGA_BLCA/fold_0.json" \
		--linear_transform_dir "../artifact/slide_embedding/${model}_linear_transform" \
		--seed 12345 \
		--savepath "../analysis/TCGA_BLCA/clustering/${model}_on_TCGA_BLCA/survival_analysis_cohort.png"
done
