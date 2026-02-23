import logging
import random

import dspy
from dspy.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class IrisDataset(Dataset):
    """
    Iris flower classification dataset (150 examples: 50 per class).

    Wraps scikit-learn's built-in Iris dataset into DSPy Examples with fields:
      sepal_length, sepal_width, petal_length, petal_width  →  answer

    Splits: 50 train / 25 dev / 75 test (shuffled with a fixed seed).
    """

    def __init__(self, seed: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from sklearn import datasets as sk_datasets

        iris = sk_datasets.load_iris()
        target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

        examples = [
            dspy.Example(
                sepal_length=float(features[0]),
                sepal_width=float(features[1]),
                petal_length=float(features[2]),
                petal_width=float(features[3]),
                answer=target_names[label].lower(),
            )
            for features, label in zip(iris.data, iris.target)
        ]

        rng = random.Random(seed)
        rng.shuffle(examples)

        examples = [
            x.with_inputs("sepal_length", "sepal_width", "petal_length", "petal_width")
            for x in examples
        ]

        self._train = examples[:50]
        self._dev   = examples[50:75]
        self._test  = examples[75:]

        logger.info(
            f"IrisDataset loaded (seed={seed}): "
            f"{len(self._train)} train / {len(self._dev)} dev / {len(self._test)} test"
        )

    def get_data_splits(self):
        return self._train, self._dev, self._test
