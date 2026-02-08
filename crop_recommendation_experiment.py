"""
Crop Recommendation experiment — ClusterFewshot PoC.

Single-flow script: Train → Evaluate → Interactive REPL.

Usage (API model):
    python crop_recommendation_experiment.py \
        --csv-path /path/to/Crop_recommendation.csv \
        --model gemini/gemini-2.0-flash

Usage (local model via sglang):
    python crop_recommendation_experiment.py \
        --csv-path /path/to/Crop_recommendation.csv \
        --model Qwen/Qwen2.5-7B-Instruct \
        --sglang-port 7501
"""

import os
import time
import argparse
import logging

import dspy
from dspy.evaluate import Evaluate
from programs import CropRecommender
from dspy.datasets.crop_recommendation import (
    CropRecommendationDataset,
    crop_recommendation_metric,
    create_crop_numeric_encoder,
    CROP_INPUT_FIELDS,
)
from dspy.teleprompt.clusterfewshot import ClusterFewshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dspy.settings.experimental = True


def main(csv_path: str, model: str, sglang_port: int | None = None):
    # -----------------------------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------------------------
    dataset = CropRecommendationDataset(csv_path=csv_path)
    trainset, devset, testset = dataset.get_data_splits()

    logger.info(f"Train: {len(trainset)}, Dev: {len(devset)}, Test: {len(testset)}")

    # -----------------------------------------------------------------------
    # 2. Configure LM
    # -----------------------------------------------------------------------
    from dspy.clients.huggingface import HFProvider

    if sglang_port or model in dspy.clients.huggingface._HF_MODELS:
        from remote_setup.utils import assign_local_lm

        port = sglang_port or 7501
        sglang_url = f"http://localhost:{port}/v1"
        lm = assign_local_lm(
            model=model,
            api_base=sglang_url,
            provider=HFProvider(
                validation_set=devset,
                validation_metric=crop_recommendation_metric,
            ),
        )
    else:
        lm = dspy.LM(model, api_key=os.getenv("GEMINI_API_KEY"), max_tokens=4000)
        dspy.configure(lm=lm)

    # -----------------------------------------------------------------------
    # 3. Set up student + ClusterFewshot optimizer
    # -----------------------------------------------------------------------
    student = CropRecommender()

    semantic_encoders = [create_crop_numeric_encoder()]

    optimizer = ClusterFewshot(
        metric=crop_recommendation_metric,
        task_type="classification",
        semantic_encoders=semantic_encoders,
        apply_visuals=True,
    )

    # -----------------------------------------------------------------------
    # 4. Compile (train)
    # -----------------------------------------------------------------------
    logger.info("Starting ClusterFewshot compilation...")
    start_time = time.time()

    optimized_program = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=devset,
    )

    elapsed = time.time() - start_time
    logger.info(f"Compilation finished in {elapsed:.1f}s")

    # Report selected demonstrations
    for name, predictor in optimized_program.named_predictors():
        logger.info(f"Predictor '{name}': {len(predictor.demos)} demos selected")

    # -----------------------------------------------------------------------
    # 5. Evaluate on test set
    # -----------------------------------------------------------------------
    evaluator = Evaluate(
        devset=testset,
        metric=crop_recommendation_metric,
        num_threads=8,
        display_progress=True,
        display_table=False,
    )

    accuracy = evaluator(optimized_program)
    logger.info(f"Test accuracy: {accuracy:.2f}%")

    # -----------------------------------------------------------------------
    # 6. Interactive REPL
    # -----------------------------------------------------------------------
    field_labels = {
        'nitrogen': 'Nitrogen (N) mg/kg',
        'phosphorous': 'Phosphorous (P) mg/kg',
        'potassium': 'Potassium (K) mg/kg',
        'temperature': 'Temperature °C',
        'humidity': 'Humidity %',
        'ph': 'Soil pH',
        'rainfall': 'Rainfall mm',
    }

    print("\n" + "=" * 60)
    print("  Crop Recommendation Agronomist")
    print("  Enter field conditions when prompted, or 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            values = {}
            for field in CROP_INPUT_FIELDS:
                raw = input(f"  {field_labels[field]}: ").strip()
                if raw.lower() == "quit":
                    print("Goodbye!")
                    return
                values[field] = float(raw)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except ValueError:
            print("  Invalid number, try again.\n")
            continue

        result = optimized_program(**values)
        print(f"\n  Recommended crop : {result.crop}")
        print(f"  Reasoning        : {result.reasoning}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Recommendation — ClusterFewshot PoC")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to Crop_recommendation.csv")
    parser.add_argument("--model", type=str, required=True, help="LM model name (e.g. gemini/gemini-2.0-flash)")
    parser.add_argument("--sglang-port", type=int, default=None,
                        help="sglang server port for local models (e.g. 7501)")
    args = parser.parse_args()

    main(args.csv_path, args.model, sglang_port=args.sglang_port)
