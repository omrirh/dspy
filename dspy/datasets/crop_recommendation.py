import random
import logging

import numpy as np
import pandas as pd

import dspy
from dspy.datasets.dataset import Dataset
from dspy.teleprompt.clusterfewshot.semantic_encoder import SemanticEncoder

logger = logging.getLogger(__name__)

# Top 5 features selected by Random Forest (pre-computed)
# Features: rainfall, humidity, potassium, phosphorous, nitrogen
CROP_INPUT_FIELDS = [
    'rainfall', 'humidity', 'potassium', 'phosphorous', 'nitrogen',
]


class CropRecommendationDataset(Dataset):
    """
    Kaggle Crop Recommendation dataset (22 crops, 5 numeric features, 2200 rows).

    Uses pre-selected top 5 features: rainfall, humidity, potassium, phosphorous, nitrogen.
    Each example carries these 5 numeric input fields and a crop label, following the
    same pattern as IrisDataset — raw features as InputFields, no text formatting.

    Source: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    """

    def __init__(self, csv_path: str, train_size: int = 500, dev_size: int = 350, test_size: int = 350, *args, **kwargs):
        """
        Args:
            csv_path: Path to the crop recommendation CSV file (use Crop_recommendation_5features.csv)
            train_size: Number of training examples
            dev_size: Number of dev examples
            test_size: Number of test examples
        """
        super().__init__(*args, **kwargs)

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path} with columns: {list(df.columns)}")

        examples = []
        for _, row in df.iterrows():
            example = dspy.Example(
                rainfall=float(row['rainfall']),
                humidity=float(row['humidity']),
                potassium=float(row['potassium']),
                phosphorous=float(row['phosphorous']),
                nitrogen=float(row['nitrogen']),
                crop=str(row['label']).strip().lower(),
            ).with_inputs(*CROP_INPUT_FIELDS)
            examples.append(example)

        import time
        seed = int(time.time())
        logger.info(f"CropRecommendation dataset seed: {seed}")

        random.seed(seed)
        random.shuffle(examples)

        total_requested = train_size + dev_size + test_size
        if total_requested > len(examples):
            raise ValueError(
                f"Requested {total_requested} examples (train={train_size}, dev={dev_size}, "
                f"test={test_size}) but only {len(examples)} available."
            )

        self._train = examples[:train_size]
        self._dev = examples[train_size:train_size + dev_size]
        self._test = examples[train_size + dev_size:train_size + dev_size + test_size]

        logger.info(
            f"Splits: {len(self._train)} train, {len(self._dev)} dev, {len(self._test)} test"
        )

    def get_data_splits(self):
        return self._train, self._dev, self._test


def _normalize_crop(name: str) -> str:
    """Normalize crop name: lowercase, strip whitespace/punctuation, remove spaces."""
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def crop_recommendation_metric(gold, pred, trace=None):
    """Case-insensitive exact match on the crop field (space/hyphen tolerant)."""
    return _normalize_crop(gold.crop) == _normalize_crop(pred.crop)


# ---------------------------------------------------------------------------
# Custom SemanticEncoder for ClusterFewshot
# ---------------------------------------------------------------------------

def crop_numeric_transform(encoder, examples):
    """
    Extract the 5 agronomic features (rainfall, humidity, potassium, phosphorous, nitrogen)
    as embedding vectors for clustering.

    This encoder uses ONLY the numeric input features, NOT the crop label.
    """
    return np.array([
        [float(ex[field]) for field in CROP_INPUT_FIELDS]
        for ex in examples
    ])


def create_crop_numeric_encoder() -> SemanticEncoder:
    """
    Factory for a numeric encoder that uses soil/environmental features only (no label).

    This encoder clusters examples based purely on agronomic similarity of the input features,
    without considering the crop label.
    """
    return SemanticEncoder(
        encoder=None,
        transform_fn=crop_numeric_transform,
        name="CropNumericEncoder",
    )
