import random
from dspy.datasets.dataset import Dataset
from sklearn import datasets


class IrisDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import dspy

        # Load the Iris dataset
        iris = datasets.load_iris()
        feature_names = iris.feature_names  # ['sepal length', 'sepal width', 'petal length', 'petal width']
        target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

        # Prepare examples formatted as DSPy expects
        examples = [
            dspy.Example(
                sepal_length=features[0],
                sepal_width=features[1],
                petal_length=features[2],
                petal_width=features[3],
                answer=target_names[label].lower(),  # Normalize label for exact match
            )
            for features, label in zip(iris.data, iris.target)
        ]

        # Shuffle deterministically
        random.seed(self.train_seed)
        random.shuffle(examples)

        # Split into 50/50/50 for train/dev/test
        self._train = examples[:50]
        self._dev = examples[50:100]
        self._test = examples[100:]

        # Sub-sample non-overlapping sets for prompt optimization
        self.train_prompt_opt = self._train[:15]  # 15 examples for prompt optimization
        self.dev_prompt_opt = self._dev[:35]  # 35 examples for validation

    def get_data_splits(self):
        """Return the dataset splits as reported in BetterTogether paper."""
        return self.train_prompt_opt, self.dev_prompt_opt, self._test
