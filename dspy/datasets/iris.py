import random
from dspy.datasets.dataset import Dataset
from sklearn import datasets


class IrisDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import dspy

        iris = datasets.load_iris()
        target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

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

        iris_examples = [x.with_inputs(
            'sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width'
        )
            for x in examples]

        self._train = iris_examples[:50]
        self._dev = iris_examples[50:100]
        self._test = iris_examples[100:150]

    def get_data_splits(self):
        return self._train, self._dev, self._test
