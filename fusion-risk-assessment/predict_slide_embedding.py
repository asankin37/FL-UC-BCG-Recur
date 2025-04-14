"""Predict embedding of slides from models of tumor stage and grade."""


import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.utils import get_simple_loader
from utils.eval_utils import initiate_model
from typing import Any


DEFAULT_ARGS = argparse.Namespace(
	drop_out=0.25,
	n_classes=2,
	model_type="clam_sb",
	model_size="small",
	embed_dim=1024
)
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def extract_embedding(
	model: Any, data: Any, hidden_state_as_embedding: bool = False
) -> np.ndarray:
	"""Return embedding of slide during CLAM prediction."""
	with torch.no_grad():
		_, _, _, _, results = model(data, return_features=True)
		feature = results["features"].cpu().numpy().flatten()
	# Optionally proceed to the final linear layer and extract the
	# intermediate hidden state as the embedding
	if hidden_state_as_embedding:
		weight = model.classifiers.weight.detach().cpu().numpy()
		bias = model.classifiers.bias.detach().cpu().numpy()
		feature = (
			feature[None, :] * weight + (bias[:, None] / feature.shape[0])
		).flatten()
	return feature


def main(
	tumor_stage_ckptpath: str,
	tumor_grade_ckptpath: str,
	patch_featdir: str,
	metadatapath: str,
	labels: list[str],
	label_col: str,
	splitpath: str | None,
	splitgroup: str | None,
	savedir: str,
	device: str,
	nested_ckpt: bool,
	hidden_state_as_embedding: bool
) -> None:
	"""
	Save predicted embedding of slides. Specifically, the array of embedding is
	stored as the `features.npy` file in the `savedir` directory, and the
	corresponding slide IDs is stored in the `slides.csv` file.
	"""
	# Load models of tumor stage and grade
	model_args = DEFAULT_ARGS
	model_args.nested_ckpt = nested_ckpt
	stage_model = initiate_model(
		model_args, ckpt_path=tumor_stage_ckptpath, device=device
	)
	grade_model = initiate_model(
		model_args, ckpt_path=tumor_grade_ckptpath, device=device
	)
	# Create dataset of slides for prediction
	dataset = Generic_MIL_Dataset(
		data_dir=patch_featdir,
		csv_path=metadatapath,
		label_dict={lbl: ind for ind, lbl in enumerate(labels)},
		label_col=label_col
	)
	if splitpath is not None and splitgroup is not None:
		dataset = dataset.return_splits(
			from_id=False, csv_path=splitpath
		)[SPLIT_INDEX[splitgroup]]
	# Predict embedding of slides and concatenate prediction from the tumor
	# stage and grade models
	features = []
	for data, _ in tqdm(get_simple_loader(dataset)):
		data = data.to(device)
		features.append(
			np.concatenate(
				[
					extract_embedding(
						stage_model, data, hidden_state_as_embedding
					),
					extract_embedding(
						grade_model, data, hidden_state_as_embedding
					)
				],
				axis=0
			)
		)
	# Save IDs and embeddings of slides
	features = np.array(features)
	with open(os.path.join(savedir, "features.npy"), "wb") as fp:
		np.save(fp, features)
	slide_inds = dataset.slide_data["slide_id"]
	slide_inds.to_frame().to_csv(os.path.join(savedir, "slides.csv"), index=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--tumor_stage_ckptpath",
		type=str,
		required=True,
		help="File path to checkpoint of the tumor stage model"
	)
	parser.add_argument(
		"--tumor_grade_ckptpath",
		type=str,
		required=True,
		help="File path to checkpoint of the tumor grade model"
	)
	parser.add_argument(
		"--patch_featdir",
		type=str,
		required=True,
		help="Directory containing the extracted features of patches"
	)
	parser.add_argument(
		"--metadatapath",
		type=str,
		required=True,
		help="File path to metadata of case IDs, slide IDs, and their labels"
	)
	parser.add_argument(
		"--labels",
		type=str,
		nargs="+",
		help=(
			"Sequence of possible labels of slides, which should match the "
			"content of the metadata"
		)
	)
	parser.add_argument(
		"--label_col",
		type=str,
		default="label",
		help="Column name of labeling in metadata"
	)
	parser.add_argument(
		"--splitpath",
		type=str,
		default=None,
		help="File path to split assignment"
	)
	parser.add_argument(
		"--splitgroup",
		type=str,
		default=None,
		choices=["train", "val", "test", None],
		help="Splitting group of which slides' embedding will be predicted"
	)
	parser.add_argument(
		"--savedir",
		type=str,
		required=True,
		help="Direcotry to save the predicted embedding of slides"
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Device to use for model prediction"
	)
	parser.add_argument(
		 "--nested_ckpt",
		 default=False,
		 action="store_true",
		 help="checkpoint is stored in a nested way by nvflare"
	)
	parser.add_argument(
		 "--hidden_state_as_embedding",
		 default=False,
		 action="store_true",
		 help="take the hidden state in the final linear layer as embedding"
	)
	args = parser.parse_args()
	os.makedirs(args.savedir, exist_ok=True)
	main(
		tumor_stage_ckptpath=args.tumor_stage_ckptpath,
		tumor_grade_ckptpath=args.tumor_grade_ckptpath,
		patch_featdir=args.patch_featdir,
		metadatapath=args.metadatapath,
		labels=args.labels,
		label_col=args.label_col,
		splitpath=args.splitpath,
		splitgroup=args.splitgroup,
		savedir=args.savedir,
		device=args.device,
		nested_ckpt=args.nested_ckpt,
		hidden_state_as_embedding=args.hidden_state_as_embedding
	)

