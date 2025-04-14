"""Cluster slides by their embedding predicted by CLAM models."""


import argparse
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import BayesianGaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingLinearTransform(BaseEstimator, TransformerMixin):
    """
    Custom transformer applying linear transformation with learned weights and
    biases.
    """

    def __init__(self, weightpath, biaspath):
        super().__init__()
        self.weight = np.load(weightpath)
        _size = self.weight.shape[-1] // 2
        self.bias = np.load(biaspath).repeat(_size) / _size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return (
            X.reshape((*X.shape[:-1], 1, X.shape[-1])) * self.weight
        ).reshape(*X.shape[:-1], -1) + self.bias


def _build_predictor(
    weightpath: str, biaspath: str, seed: int | None = None
) -> tuple[Pipeline, BayesianGaussianMixture]:
    return (
        make_pipeline(
            EmbeddingLinearTransform(weightpath, biaspath),
            FeatureAgglomeration(
                n_clusters=512,
                linkage="complete",
                pooling_func=np.sum,
                connectivity=lambda x: kneighbors_graph(x, n_neighbors=20)
            ),
            Normalizer(norm="max"),
            FeatureAgglomeration(n_clusters=128, linkage="average", metric="l1"),
        ),
        BayesianGaussianMixture(
            n_components=2, covariance_type="diag", random_state=seed
        )
    )


def main(
    embdpath_fit: str,
    embdpath_pred: str,
    slidepath_pred: str,
    weightpath: str,
    biaspath: str,
    seed: int | None,
    savedir: str
) -> None:
    """
    Save the cluster assignment of slides, their log-likelihood for each
    cluster, and their embedding standardized for each cluster.
    """
    embd_fit = np.load(embdpath_fit)
    embd_pred = np.load(embdpath_pred)
    preprocessor, mixture = _build_predictor(
        weightpath=weightpath, biaspath=biaspath, seed=seed
    )
    preprocessor = preprocessor.fit(embd_fit)
    mixture = mixture.fit(preprocessor.transform(embd_fit))
    # Save cluster assignment of slides
    clustering = pd.DataFrame(
        {
            "slide_id": pd.read_csv(slidepath_pred)["slide_id"],
            "cluster_id": mixture.predict(preprocessor.transform(embd_pred))
        }
    )
    clustering.to_csv(os.path.join(savedir, "cluster.csv"), index=False)
    # Save log likelihood for each cluster
    log_likelihood = mixture._estimate_weighted_log_prob(
        preprocessor.transform(embd_pred)
    )
    with open(os.path.join(savedir, "log_likelihood.npy"), "wb") as fp:
        np.save(fp, log_likelihood)
    # Save log likelihood for training samples
    with open(os.path.join(savedir, "log_likelihood_fit.npy"), "wb") as fp:
        np.save(
            fp,
            mixture._estimate_weighted_log_prob(preprocessor.transform(embd_fit))
        )
    # Save processed embedding for each cluster
    preprocessed_embedding = preprocessor.transform(embd_pred)
    with open(os.path.join(savedir, "preprocessed_embedding.npy"), "wb") as fp:
        np.save(fp, preprocessed_embedding)
    # Save parameters of the mixture model
    with open(os.path.join(savedir, "means.npy"), "wb") as fp:
        np.save(fp, mixture.means_)
    with open(os.path.join(savedir, "covariances.npy"), "wb") as fp:
        np.save(fp, mixture.covariances_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide_featdir_fit",
        type=str,
        required=True,
        help="Directory containing the IDs and embedding of slides for fitting"
    )
    parser.add_argument(
        "--slide_featdir_pred",
        type=str,
        required=True,
        help="Directory containing the IDs and embedding of slides to cluster"
    )
    parser.add_argument(
        "--linear_transform_dir",
        type=str,
        required=True,
        help="Dictionary containing learned parameter of a linear layer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seeding of random number generator"
    )
    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help="Directory path to save clustering prediction of slides"
    )
    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    main(
        embdpath_fit=os.path.join(args.slide_featdir_fit, "features.npy"),
        embdpath_pred=os.path.join(args.slide_featdir_pred, "features.npy"),
        slidepath_pred=os.path.join(args.slide_featdir_pred, "slides.csv"),
        weightpath=os.path.join(args.linear_transform_dir, "weight.npy"),
        biaspath=os.path.join(args.linear_transform_dir, "bias.npy"),
        seed=args.seed,
	savedir=args.savedir
    )

