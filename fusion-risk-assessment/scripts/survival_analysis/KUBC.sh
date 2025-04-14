#!/bin/bash

set -e


python log_rank_test.py \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--survivalpath "../external/relapse_free_survival.csv" \
	--clusterdir "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--riskgroupdir "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC" \
	--num_folds 1 \
	--savepath "../analysis/survival_analysis/FL_foldless_on_KUBC/log_rank_test.csv"

python visualize_kaplan_meier.py \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--survivalpath "../external/relapse_free_survival.csv" \
	--clusterdir "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--riskgroupdir "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC" \
	--num_folds 1 \
	--testrespath "../analysis/survival_analysis/FL_foldless_on_KUBC/log_rank_test.csv" \
	--savepath "../analysis/survival_analysis/FL_foldless_on_KUBC/kaplan_meier_plot.png"

python univariate_cox_regression.py \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--survivalpath "../external/relapse_free_survival.csv" \
	--clusterdir "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--riskgroupdir "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC" \
	--num_folds 1 \
	--additional_metadatapath "../external/additional_metadata_survival_analysis_cohort.csv" \
	--additional_features "Positive reTUR" \
	--savepath "../analysis/survival_analysis/FL_foldless_on_KUBC/univariate_cox_regression.csv"

python multivariate_cox_regression.py \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--survivalpath "../external/relapse_free_survival.csv" \
	--clusterdir "../artifact/clustering/survival_analysis_cohort/FL_foldless_on_KUBC" \
	--riskgroupdir "../artifact/clustering/risk_group_mapping/FL_foldless_on_KUBC" \
	--num_folds 1 \
	--additional_metadatapath "../external/additional_metadata_survival_analysis_cohort.csv" \
	--additional_features "Positive reTUR" \
	--savedir "../analysis/survival_analysis/FL_foldless_on_KUBC" \
	--saveprefix "multivariate_cox_regression"
