#!/bin/bash

set -e

# Slides in the CLAM cohort
python predict_slide_embedding.py \
	--tumor_stage_ckptpath "../model/FL_uc_stage_foldless/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/FL_uc_grade_foldless/s_0_checkpoint.pt" \
	--patch_featdir "../artifact/patches" \
	--metadatapath "../artifact/metadata_clam_cohort.csv" \
	--labels "low-grade" "high-grade" \
	--label_col "uc_grade" \
	--savedir "../artifact/slide_embedding/clam_cohort/FL_foldless_on_VU" \
	--device "cuda" \
	--nested_ckpt

# Slides in the survival analysis cohort
python predict_slide_embedding.py \
	--tumor_stage_ckptpath "../model/FL_uc_stage_foldless/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/FL_uc_grade_foldless/s_0_checkpoint.pt" \
	--patch_featdir "../artifact/patches" \
	--metadatapath "../artifact/metadata_survival_analysis_cohort.csv" \
	--labels "low-grade" "high-grade" \
	--label_col "uc_grade" \
	--savedir "../artifact/slide_embedding/survival_analysis_cohort/FL_foldless_on_VU" \
	--device "cuda" \
	--nested_ckpt
