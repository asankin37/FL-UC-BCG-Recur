#!/bin/bash

set -e

# Slide embedding extracted with the FL model
python predict_slide_embedding.py \
	--tumor_stage_ckptpath "../model/FL_uc_stage_foldless/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/FL_uc_grade_foldless/s_0_checkpoint.pt" \
	--patch_featdir "../artifact/TCGA_BLCA/patches" \
	--metadatapath "../artifact/TCGA_BLCA/metadata_clam_cohort.csv" \
	--labels "low-grade" "high-grade" \
	--label_col "uc_grade" \
	--savedir "../artifact/TCGA_BLCA/slide_embedding/clam_cohort/FL_foldless_on_TCGA_BLCA" \
	--device "cuda" \
	--nested_ckpt

python predict_slide_embedding.py \
	--tumor_stage_ckptpath "../model/FL_uc_stage_foldless/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/FL_uc_grade_foldless/s_0_checkpoint.pt" \
	--patch_featdir "../artifact/TCGA_BLCA/patches" \
	--metadatapath "../artifact/TCGA_BLCA/metadata_survival_analysis_cohort.csv" \
	--labels "low-grade" "high-grade" \
	--label_col "uc_grade" \
	--savedir "../artifact/TCGA_BLCA/slide_embedding/survival_analysis_cohort/FL_foldless_on_TCGA_BLCA" \
	--device "cuda" \
	--nested_ckpt
