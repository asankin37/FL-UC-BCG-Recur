#!/bin/bash

set -e

python get_model_linear_transform.py \
	--tumor_stage_ckptpath "../model/KUBC_uc_stage/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/KUBC_uc_grade/s_0_checkpoint.pt" \
	--savedir "../artifact/slide_embedding/KUBC_linear_transform" \
	--device "cuda"

python get_model_linear_transform.py \
	--tumor_stage_ckptpath "../model/VU_uc_stage/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/VU_uc_grade/s_0_checkpoint.pt" \
	--savedir "../artifact/slide_embedding/VU_linear_transform" \
	--device "cuda"

python get_model_linear_transform.py \
	--tumor_stage_ckptpath "../model/FL_uc_stage_foldless/s_0_checkpoint.pt" \
	--tumor_grade_ckptpath "../model/FL_uc_grade_foldless/s_0_checkpoint.pt" \
	--savedir "../artifact/slide_embedding/FL_foldless_linear_transform" \
	--device "cuda" \
	--nested_ckpt
