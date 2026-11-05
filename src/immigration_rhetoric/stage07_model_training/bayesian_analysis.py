"""Quantify the uncertainty in model performance across training runs on different train/eval splits.

Each metric's F1 scores (one per split) are modelled as draws from a Normal distribution whose mean has a Beta
prior and whose standard deviation has a HalfNormal prior. The posterior of each mean gives an estimate of that
metric with a credible interval.
"""

import json
import logging
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.link.c.cmodule as cmodule
from immigration_rhetoric.common.logging_utils import TqdmLoggingHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[TqdmLoggingHandler()],
)

if sys.platform == "darwin":
    # PyTensor passes -ld64 to the linker on macOS 15+, which newer Xcode linkers reject
    _compile_args = cmodule.GCC_compiler.compile_args
    cmodule.GCC_compiler.compile_args = staticmethod(
        lambda *args, **kwargs: [
            flag for flag in _compile_args(*args, **kwargs) if flag != "-ld64"
        ]
    )

# Where each metric is read from: (model folder, path of keys within its metrics.json)
METRIC_SOURCES = {
    "overall": ("combined", ["cats_macro_f"]),
    "positive": ("combined", ["cats_f_per_type", "POSITIVE_TOWARDS_IMMIGRATION", "f"]),
    "relevance": ("relevance", ["cats_macro_f"]),
    "sentiment": ("sentiment", ["cats_macro_f"]),
    "neutral": ("combined", ["cats_f_per_type", "NEUTRAL_TOWARDS_IMMIGRATION", "f"]),
    "negative": ("combined", ["cats_f_per_type", "NEGATIVE_TOWARDS_IMMIGRATION", "f"]),
    "not_about": ("combined", ["cats_f_per_type", "NOT_ABOUT_IMMIGRATION", "f"]),
}

# Based on earlier training runs. mu/sigma: mean and sd of the Beta prior on each metric's mean F1;
# tau: scale of the HalfNormal prior on its run-to-run sd.
# The order sets the order of the model's variables, which changes the seeded samples - don't reorder.
PRIORS = {
    "overall": {"mu": 0.65, "sigma": 0.05, "tau": 0.01},
    "positive": {"mu": 0.65, "sigma": 0.05, "tau": 0.005},
    "relevance": {"mu": 0.89, "sigma": 0.05, "tau": 0.01},
    "sentiment": {"mu": 0.66, "sigma": 0.05, "tau": 0.01},
    "neutral": {"mu": 0.55, "sigma": 0.05, "tau": 0.015},
    "negative": {"mu": 0.65, "sigma": 0.05, "tau": 0.005},
    "not_about": {"mu": 0.85, "sigma": 0.05, "tau": 0.01},
}

SUMMARY_ORDER = [
    "overall",
    "relevance",
    "sentiment",
    "positive",
    "neutral",
    "negative",
    "not_about",
]


def beta_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    """Convert the mean and standard deviation of a Beta distribution to its alpha and beta parameters."""
    tmp = mean * (1 - mean) / sd**2 - 1
    return mean * tmp, (1 - mean) * tmp


class PerformanceUncertaintyEstimator:
    def __init__(
        self,
        priors: dict[str, dict[str, float]] = PRIORS,
        draws: int = 4000,
        tune: int = 2000,
        chains: int = 4,
        target_accept: float = 0.95,
        random_seed: int = 42,
    ):
        self.priors = priors
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed

    @staticmethod
    def load_f1_scores(training_dir: Path, seeds: list[int]) -> dict[str, list[float]]:
        """Read each metric's F1 score for every seed from `{training_dir}/rs{seed}/{model}/metrics.json`."""
        f1_scores = {name: [] for name in METRIC_SOURCES}
        for seed in seeds:
            metrics = {
                model: json.loads(
                    (training_dir / f"rs{seed}" / model / "metrics.json").read_text()
                )
                for model in ("relevance", "sentiment", "combined")
            }
            for name, (model, keys) in METRIC_SOURCES.items():
                value = metrics[model]
                for key in keys:
                    value = value[key]
                f1_scores[name].append(value)
        logging.info(f"Loaded F1 scores for seeds {seeds}:")
        for name, scores in f1_scores.items():
            logging.info(f"  {name}: {', '.join(f'{score:.4f}' for score in scores)}")
        return f1_scores

    def fit(self, f1_scores: dict[str, list[float]]) -> az.InferenceData:
        """Fit the model to the F1 scores and return the posterior samples."""
        logging.info(
            f"Sampling posterior ({self.chains} chains, {self.draws} draws, {self.tune} tuning steps)..."
        )
        with pm.Model():
            for name, prior in self.priors.items():
                alpha, beta = beta_from_mean_sd(prior["mu"], prior["sigma"])
                mu = pm.Beta(
                    f"mu_{name}", alpha=np.round(alpha, 1), beta=np.round(beta, 1)
                )
                sigma = pm.HalfNormal(f"sigma_{name}", sigma=prior["tau"])
                pm.Normal(
                    f"f1_{name}", mu=mu, sigma=sigma, observed=np.array(f1_scores[name])
                )

            return pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
            )

    @staticmethod
    def summarise(trace: az.InferenceData, hdi_prob: float = 0.95) -> pd.DataFrame:
        """Summarise the posterior of each metric's mean F1, with its highest density interval and diagnostics."""
        return az.summary(
            trace,
            var_names=[f"mu_{name}" for name in SUMMARY_ORDER],
            hdi_prob=hdi_prob,
            round_to=3,
        )
