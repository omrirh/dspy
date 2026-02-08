# CropRecommender: LLM-as-an-Agronomist via ClusterFewshot

An external use-case showcasing how ClusterFewshot derives quality few-shot demonstrations for an LLM-based crop recommendation system.

## Overview

Given a field observation — soil nutrient levels (N, P, K) and environmental conditions (temperature, humidity, pH, rainfall) — the system recommends the single most suitable crop to cultivate, with expert agronomic reasoning.

**Data source:** [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data) (2200 rows, 22 crops, 7 numeric features, Apache 2.0 license).

## Architecture

```
                          +-----------------------+
                          |  CropRecommendation   |
                          |      Dataset          |
                          |  (CSV -> Examples)    |
                          +----------+------------+
                                     |
                    trainset         |         devset
                   +--------+--------+--------+--------+
                   |                                   |
          +--------v--------+               +----------v---------+
          |  CropNumeric    |               |   ClusterFewshot   |
          |  Encoder        |               |   Optimizer        |
          |  (7-dim vects)  |               |                    |
          +--------+--------+               +----------+---------+
                   |                                   |
                   +----------> K-means <--------------+
                              Clustering
                                  |
                          Agronomic Zones
                          (latent clusters)
                                  |
                     +------------+-------------+
                     |                          |
              One-Shot Ranking          best_in_cluster
              (per-demo eval)            Sampling
                     |                          |
                     +------------+-------------+
                                  |
                        Optimized Few-Shot Set
                                  |
                     +------------v-------------+
                     |    CropRecommender       |
                     |    (ChainOfThought)      |
                     |  observation -> crop     |
                     +--------------------------+
```

## Design Decisions

### 1. Example Schema: Dual Representation

Each `dspy.Example` carries both a **textual observation** (for the LLM) and **raw numeric fields** (for the encoder):

```python
dspy.Example(
    # Input for the LLM program
    observation="Soil Analysis - Nitrogen (N): 90 mg/kg, Phosphorous (P): 42 mg/kg, ...",

    # Auxiliary numeric fields for the encoder (not marked as inputs)
    nitrogen=90.0, phosphorous=42.0, potassium=43.0,
    temperature=20.87, humidity=82.0, ph=6.5, rainfall=202.93,

    # Output label
    crop="rice",
).with_inputs('observation')
```

**Why this works:** ClusterFewshot's encoder accesses raw example fields by name (not via `.inputs()`), so the numeric fields are available for clustering even though only `observation` is the program input. Bootstrapped demos capture the predictor trace (`observation -> reasoning, crop`), so numeric fields never leak into LLM prompts.

### 2. CropNumericEncoder: Agronomic Feature Space

The custom encoder extracts 7 physical measurements as embedding vectors:

```python
# nitrogen=90, phosphorous=42, potassium=43, temperature=20.87,
# humidity=82.0, ph=6.5, rainfall=202.93
# -> embedding: [90.0, 42.0, 43.0, 20.87, 82.0, 6.5, 202.93]
```

When ClusterFewshot runs K-means on these 7-dimensional vectors, it discovers **latent agronomic zones** — groups of field conditions that share similar growing requirements:

| Zone | Conditions | Typical Crops |
|------|-----------|---------------|
| Paddy/Wetland | High rainfall, moderate temp, acidic soil | rice, jute |
| Arid-tolerant | Low humidity, high temp, alkaline soil | mothbeans, mungbean |
| Fruit-growing | High potassium, moderate rain, mild temp | mango, grapes, apple |
| Legume belt | Balanced NPK, tropical conditions | lentil, chickpea |

The numeric encoder is natural here because **agronomic similarity IS defined by these measurements**. Two fields with similar N/P/K/temperature/humidity/pH/rainfall will support similar crops.

### 3. Why Clustering Yields Better Expert Advice

Without clustering, selecting the top-N globally-ranked demonstrations risks **agronomic blind spots**: the best-scoring examples might all come from similar conditions (e.g., all tropical grain scenarios), leaving the LLM without guidance for arid, temperate, or fruit-growing queries.

ClusterFewshot's `best_in_cluster` sampling (used for classification tasks) ensures the few-shot set **covers all discovered agronomic zones**:

1. **Diversity via cluster coverage:** One demo per zone guarantees the LLM sees examples across the full range of growing conditions.
2. **Quality via one-shot ranking:** Each training example is tested as a standalone demo on validation data. Examples that help the LLM predict correctly across many conditions rank higher — they're better "teachers."
3. **The latent task:** Each cluster represents a distinct agronomic decision context. At inference time, whatever new observation arrives will fall near one of these zones, ensuring the LLM always has a relevant precedent.

### 4. CropRecommender Program

Uses `dspy.ChainOfThought` which automatically generates reasoning before the crop prediction:

```python
class CropRecommenderSignature(dspy.Signature):
    """You are an expert agronomist advisor. Given a field observation describing
    soil nutrient levels and environmental conditions, recommend the single most
    suitable crop to cultivate."""

    observation = dspy.InputField(desc="Soil nutrient analysis (N, P, K) and environmental conditions")
    crop = dspy.OutputField(desc="The recommended crop name")

class CropRecommender(dspy.Module):
    def __init__(self):
        self.recommend = dspy.ChainOfThought(CropRecommenderSignature)

    def forward(self, observation):
        return self.recommend(observation=observation)
```

The `ChainOfThought` wrapper adds a `reasoning` field before `crop`, producing demonstrations like:

```
Observation: Soil Analysis - Nitrogen (N): 90 mg/kg, ...
Reasoning: The high nitrogen content and significant rainfall suggest wetland-adapted crops.
            The moderate temperature and slightly acidic pH further support paddy cultivation...
Crop: rice
```

## File Structure

| File | Purpose |
|------|---------|
| `dspy/datasets/crop_recommendation.py` | Dataset, metric, custom encoder |
| `programs.py` | `CropRecommender` program (alongside existing CoT, IrisProgram) |
| `crop_recommendation_experiment.py` | End-to-end: train -> evaluate -> interactive REPL |
| `crop_recommendation_usecase.md` | This documentation |
| `crop_recommendation_demo.ipynb` | Interactive notebook walkthrough |

## Quick Start

```bash
# Download the CSV from Kaggle, then:
python crop_recommendation_experiment.py \
    --csv-path /path/to/Crop_recommendation.csv \
    --model gemini/gemini-2.5-flash
```

The script will:
1. Load and split the dataset (70/15/15 train/dev/test)
2. Compile with ClusterFewshot (bootstrap -> cluster -> rank -> select)
3. Evaluate on the test set and report accuracy
4. Enter an interactive REPL for live crop recommendations
