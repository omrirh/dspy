# ClusterFewshot Documentation

## Overview

**ClusterFewshot** is a task-adaptive few-shot selection teleprompter that optimizes demonstrations for DSPy programs using semantic clustering and empirical evaluation.

### How It Works

ClusterFewshot follows a three-stage optimization pipeline:

1. **Semantic Clustering**: Groups training examples into semantically coherent clusters using custom encoders
2. **One-Shot Evaluation**: Ranks each example by its effectiveness as a single demonstration
3. **Task-Adaptive Selection**: Chooses the optimal few-shot subset by evaluating different sampling strategies

### Key Features

**Bring-Your-Own-Encoder (BYOE)**: Plug in custom semantic encoders tailored to your task
**Automatic Encoder Selection**: Grid search over encoders to find the best clustering
**Empirical Ranking**: Evaluates each example's actual impact as a demonstration
**Task-Adaptive Sampling**: Different strategies for arithmetic, QA, and classification tasks
**Soft Selection (Optional)**: Differentiable optimization balancing quality and diversity
**Visualization Support**: PCA plots and score distributions (optional)

---

## Installation & Imports

```python
import dspy
from dspy.teleprompt.clusterfewshot import (
    ClusterFewshot,
    SemanticEncoder,
    create_sentence_transformer_encoder,
    create_numeric_encoder,
    sentence_transformer_transform,
    numeric_transform,
)
```

---

## Bring-Your-Own-Encoder Design

### What is a SemanticEncoder?

A `SemanticEncoder` encapsulates:
- **Encoder model**: The model that generates embeddings (e.g., SentenceTransformer, LLM, or None)
- **Transform function**: How to apply the encoder to your examples
- **Name**: Identifier for logging and visualization

### Creating Encoders

#### 1. Using Factory Functions (Recommended)

**For text-based tasks (QA, arithmetic, reasoning):**
```python
encoder = create_sentence_transformer_encoder(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="cpu"  # or "cuda"
)
```

**For classification with numeric inputs:**
```python
encoder = create_numeric_encoder()
```

#### 2. Custom SemanticEncoder

```python
from sentence_transformers import SentenceTransformer

# Define your custom transform
def custom_transform(encoder, examples):
    # Extract relevant fields and encode
    texts = [f"{ex.question} {ex.context}" for ex in examples]
    return encoder.encode(texts, convert_to_numpy=True)

# Create the encoder
model = SentenceTransformer("your-model-name")
encoder = SemanticEncoder(
    encoder=model,
    transform_fn=custom_transform,
    name="CustomContextualEncoder"
)
```

#### 3. Task-Tuned LLM Encoder

```python
from transformers import AutoModel, AutoTokenizer
import torch

def llm_transform(encoder, examples):
    tokenizer, model = encoder
    texts = [ex.question for ex in examples]

    # Tokenize and get embeddings
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

    return embeddings.cpu().numpy()

tokenizer = AutoTokenizer.from_pretrained("your-llm")
model = AutoModel.from_pretrained("your-llm")

encoder = SemanticEncoder(
    encoder=(tokenizer, model),
    transform_fn=llm_transform,
    name="TaskTunedLLM"
)
```

---

## Basic Usage

### Minimal Example

```python
import dspy
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Configure DSPy
dspy.settings.experimental = True
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load dataset
dataset = GSM8K()
trainset = [x.with_inputs('question') for x in dataset.train[:50]]
valset = [x.with_inputs('question') for x in dataset.dev[50:150]]

# Define your program
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

student = CoT()

# Create semantic encoders
encoders = [
    create_sentence_transformer_encoder("all-mpnet-base-v2"),
    create_sentence_transformer_encoder("gtr-t5-base"),
]

# Initialize optimizer
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=encoders,
)

# Compile
optimized_program = optimizer.compile(
    student=student,
    trainset=trainset,
    valset=valset
)
```

---

## Task Types & Sampling Strategies

ClusterFewshot uses different sampling strategies based on task type:

### 1. Arithmetic Tasks
**Strategies**: `top_n`, `best_in_cluster`

```python
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=[...],
)
```

**Use for**: Math word problems, numerical reasoning, calculation tasks

### 2. Multi-hop QA Tasks
**Strategies**: `top_n`, `best_in_cluster`

```python
optimizer = ClusterFewshot(
    task_type="multihop",
    metric=dspy.evaluate.answer_exact_match,
    semantic_encoders=[...],
)
```

**Use for**: Complex QA requiring multiple reasoning steps, HotPotQA, etc.

### 3. Classification Tasks
**Strategies**: `best_in_cluster`

```python
optimizer = ClusterFewshot(
    task_type="classification",
    metric=dspy.evaluate.answer_exact_match,
    semantic_encoders=[create_numeric_encoder()],
)
```

**Use for**: Iris dataset, sentiment analysis, topic classification

### Sampling Strategy Details

- **top_n**: Selects globally top-ranked examples that appear in each cluster
- **best_in_cluster**: Picks the highest-ranked example from each cluster
- **popularity**: Allocates slots proportionally to cluster sizes (currently not in default strategies)
- **central**: Selects the most semantically central example per cluster (currently not in default strategies)

---

## Advanced Features

### 1. Soft Selection

Balance demonstration quality with semantic diversity using gradient descent:

```python
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=encoders,
    soft_select=True,  # Enable differentiable selection
)
```

**How it works:**
- Optimizes a probability distribution over training examples
- Maximizes one-shot impact while minimizing semantic redundancy
- Uses learnable diversity penalty (lambda) optimized jointly with selection

### 2. Visualization Control

```python
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=encoders,
    apply_visuals=False,  # Disable matplotlib visualizations
)
```

**When disabled**, no plots are generated (faster, useful for batch experiments)
**When enabled** (default), generates:
- PCA plots of training/validation clusters
- One-shot score distributions
- Embedding-to-rank correlations
- Soft selection visualizations (if soft_select=True)

### 3. Metric Threshold for Bootstrapping

Filter training examples by quality during bootstrapping:

```python
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    metric_threshold=0.8,  # Only keep examples with metric >= 0.8
    semantic_encoders=encoders,
)
```

---

## Complete Examples

### Example 1: GSM8K (Arithmetic)

```python
import dspy
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.experimental = True
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Prepare data
dataset = GSM8K()
trainset = [x.with_inputs('question') for x in dataset.train[:1000]]
valset = [x.with_inputs('question') for x in dataset.dev[:500]]

# Define program
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# Configure encoders
encoders = [
    create_sentence_transformer_encoder("Qwen/Qwen3-Embedding-0.6B"),
    create_sentence_transformer_encoder("all-mpnet-base-v2"),
]

# Optimize
optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=encoders,
)

optimized = optimizer.compile(
    student=CoT(),
    trainset=trainset,
    valset=valset
)
```

### Example 2: HotPotQA (Multi-hop)

```python
import dspy
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder
from dspy.datasets import HotPotQA

dspy.settings.experimental = True
lm = dspy.LM("openai/gpt-4o-mini")
rm = dspy.ColBERTv2(url="http://localhost:8894/api/search")  # Local retriever
dspy.configure(lm=lm, rm=rm)

# Prepare data
dataset = HotPotQA(only_hard_examples=True)
trainset = [x.with_inputs('question') for x in dataset.train[:1000]]
valset = [x.with_inputs('question') for x in dataset.dev[:500]]

# Define program
class MultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.predict = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.predict(context=context, question=question)

# Configure encoders
encoders = [
    create_sentence_transformer_encoder("gtr-t5-base"),
    create_sentence_transformer_encoder("BAAI/bge-large-en-v1.5"),
]

# Optimize
optimizer = ClusterFewshot(
    task_type="multihop",
    metric=dspy.evaluate.answer_exact_match,
    semantic_encoders=encoders,
)

optimized = optimizer.compile(
    student=MultiHop(),
    trainset=trainset,
    valset=valset
)
```

### Example 3: Iris (Classification)

```python
import dspy
from dspy.teleprompt.clusterfewshot import ClusterFewshot, create_numeric_encoder
from dspy.datasets import IrisDataset

dspy.settings.experimental = True
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Prepare data
dataset = IrisDataset()
trainset, valset, testset = dataset.get_data_splits()

# Define program
class IrisProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(
            "sepal_length, sepal_width, petal_length, petal_width -> species"
        )

    def forward(self, sepal_length, sepal_width, petal_length, petal_width):
        return self.predict(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )

# Use numeric encoder (no text encoding needed)
encoders = [create_numeric_encoder()]

# Optimize
optimizer = ClusterFewshot(
    task_type="classification",
    metric=dspy.evaluate.answer_exact_match,
    semantic_encoders=encoders,
)

optimized = optimizer.compile(
    student=IrisProgram(),
    trainset=trainset,
    valset=valset
)
```

---

## API Reference

### ClusterFewshot

```python
ClusterFewshot(
    task_type: str,
    metric: Optional[Callable] = None,
    metric_threshold: Optional[float] = None,
    soft_select: bool = False,
    apply_visuals: bool = True,
    semantic_encoders: Optional[List[SemanticEncoder]] = None
)
```

**Parameters:**
- `task_type` (str): Task type - `"arithmetic"`, `"multihop"`, or `"classification"`
- `metric` (Callable): Evaluation function `(example, prediction, trace) -> score`
- `metric_threshold` (float): Minimum score to keep examples during bootstrapping
- `soft_select` (bool): Use differentiable soft selection (default: False)
- `apply_visuals` (bool): Generate matplotlib visualizations (default: True)
- `semantic_encoders` (List[SemanticEncoder]): **Required** - List of encoders to evaluate

**Methods:**
- `compile(student, trainset, valset)`: Optimize the student program and return optimized version

### SemanticEncoder

```python
SemanticEncoder(
    encoder: Any,
    transform_fn: Callable[[Any, List[Example]], np.ndarray],
    name: Optional[str] = None
)
```

**Parameters:**
- `encoder`: The model (SentenceTransformer, LLM, or None)
- `transform_fn`: Function `(encoder, examples) -> embeddings`
- `name`: Optional encoder name for logging

**Methods:**
- `encode(examples)`: Encode examples and return embeddings array

### Factory Functions

```python
create_sentence_transformer_encoder(model_name: str, device: str = 'cpu') -> SemanticEncoder
```
Creates a SentenceTransformer-based encoder.

```python
create_numeric_encoder() -> SemanticEncoder
```
Creates a numeric/identity encoder for classification tasks, such as Iris classification.

---

## Best Practices

1. **Start with multiple encoders**: Let ClusterFewshot find the best one via grid search
2. **Match encoder to task**: Use text encoders for QA/reasoning, numeric for classification
6. **Monitor silhouette scores**: Higher = better clustering quality (check logs)

---

## Troubleshooting

### Error: "semantic_encoders parameter is required"
**Solution**: Always provide at least one encoder:
```python
semantic_encoders=[create_sentence_transformer_encoder("all-mpnet-base-v2")]
```

---

## Contributing

Contributions welcome! Areas for improvement:
- New semantic encoders for specific domains
- Additional task types and sampling strategies
- Performance optimizations
- Documentation and examples

Please submit PRs to the [DSPy repository](https://github.com/stanfordnlp/dspy).