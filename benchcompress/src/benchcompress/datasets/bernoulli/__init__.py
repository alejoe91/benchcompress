import numpy as np
import os


SOURCE_FILE = "bernoulli/__init__.py"


def _load_long_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, "bernoulli.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


LONG_DESCRIPTION = _load_long_description()


def create_bernoulli(*, n_samples: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, p, n_samples).astype(np.uint8)
    return x


tags = ["bernoulli", "timeseries", "1d", "integer", "discrete", "synthetic", "i.i.d.", "unsigned"]

datasets = [
    {
        "name": "bernoulli-0.1",
        "version": "3",
        "create": lambda: create_bernoulli(n_samples=1_000_000, p=0.1, seed=0),
        "description": "Binary sequence with 10% probability of ones.",
        "tags": tags,
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "bernoulli-0.2",
        "version": "3",
        "create": lambda: create_bernoulli(n_samples=1_000_000, p=0.2, seed=0),
        "description": "Binary sequence with 20% probability of ones.",
        "tags": tags,
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "bernoulli-0.3",
        "version": "3",
        "create": lambda: create_bernoulli(n_samples=1_000_000, p=0.3, seed=0),
        "description": "Binary sequence with 30% probability of ones.",
        "tags": tags,
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "bernoulli-0.4",
        "version": "3",
        "create": lambda: create_bernoulli(n_samples=1_000_000, p=0.4, seed=0),
        "description": "Binary sequence with 40% probability of ones.",
        "tags": tags,
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
    {
        "name": "bernoulli-0.5",
        "version": "3",
        "create": lambda: create_bernoulli(n_samples=1_000_000, p=0.5, seed=0),
        "description": "Binary sequence with 50% probability of ones and 50% probability of zeros.",
        "tags": tags,
        "source_file": SOURCE_FILE,
        "long_description": LONG_DESCRIPTION,
    },
]
