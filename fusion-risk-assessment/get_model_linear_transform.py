"""Extract parameters of the final linear layer in a trained model."""


import argparse
import os
import numpy as np
from utils.eval_utils import initiate_model


DEFAULT_ARGS = argparse.Namespace(
	drop_out=0.25,
	n_classes=2,
	model_type="clam_sb",
	model_size="small",
	embed_dim=1024
)


def main(
	tumor_stage_ckptpath: str,
	tumor_grade_ckptpath: str,
	savedir: str,
	device: str,
	nested_ckpt: bool
) -> None:
	"""
	Save the weights and biases of the final linear layer in a mode
	checkpoint.
	"""
	model_args = DEFAULT_ARGS
	model_args.nested_ckpt = nested_ckpt
	model_stage = initiate_model(
		model_args, ckpt_path=tumor_stage_ckptpath, device=device
	)
	model_grade = initiate_model(
		model_args, ckpt_path=tumor_grade_ckptpath, device=device
	)
	weight = np.concatenate(
		[
			model_stage.classifiers.weight.detach().cpu().numpy(),
			model_grade.classifiers.weight.detach().cpu().numpy()
		],
		axis=-1
	)
	bias = np.concatenate(
		[
			model_stage.classifiers.bias.detach().cpu().numpy(),
			model_grade.classifiers.bias.detach().cpu().numpy()
		],
		axis=-1
	)
	np.save(os.path.join(savedir, "weight.npy"), weight)
	np.save(os.path.join(savedir, "bias.npy"), bias)


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
		"--savedir",
		type=str,
		required=True,
		help="Directory to save weights and biases of the final linear layer"
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
	args = parser.parse_args()
	os.makedirs(args.savedir, exist_ok=True)
	main(
		tumor_stage_ckptpath=args.tumor_stage_ckptpath,
		tumor_grade_ckptpath=args.tumor_grade_ckptpath,
		savedir=args.savedir,
		device=args.device,
		nested_ckpt=args.nested_ckpt
	)
