import numpy as np
import dspy
from scipy.stats import entropy
from dspy.evaluate import Evaluate


class EvaluateEntropy(Evaluate):
    """Evaluates the certainty of predictions by summing the entropies of a program's predictions on a devset.

    - Correct predictions: Sum entropy as usual (confidence measure).
    - Incorrect predictions: Add a penalty by summing a max entropy value.
    """

    def __init__(self, *, devset, penalty_factor=2.0, **kwargs):
        super().__init__(devset=devset, **kwargs)
        self.penalty_factor = penalty_factor

    def __call__(self, program: "dspy.Module", **kwargs):
        """
        Runs the evaluation and calculates the sum of entropies of the predictions.

        Returns:
            float: The total entropy across all predictions, with penalties for incorrect ones.
        """

        # Run the base evaluation method to get the predictions
        score, results = super().__call__(program, return_outputs=True, **kwargs)

        total_entropy = 0.0
        for example, prediction, success in results:
            if hasattr(prediction, 'logits'):
                # TODO: nope, need to attach logprobs to prediction instance. see:
                #  https://github.com/omrirh/dspy/blob/281bf2e77c1910a70b9b8f63642cb4ab159f2f17/dspy/clients/lm.py#L129
                probs = softmax(prediction.logits)
                pred_entropy = entropy(probs, base=2)

                if success:  # Correct prediction
                    total_entropy += pred_entropy
                else:  # Incorrect prediction, apply penalty
                    total_entropy += self.penalty_factor * pred_entropy
            else:
                raise ValueError("Prediction object must contain 'logits' for entropy calculation.")

        return total_entropy


def softmax(logits):
    """Computes softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)
