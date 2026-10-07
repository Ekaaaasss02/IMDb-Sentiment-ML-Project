import matplotlib

# Use a non-GUI backend to prevent Tkinter/Tcl errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


# Allow `python src/train.py` to import sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import clean_corpus
from models import build_model
from evaluate import evaluate_model


# All models used in the project
ALL_MODELS = [
    "logistic",
    "svm",
    "naive_bayes",
    "random_forest",
    "boosting",
    "dnn",
]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the IMDb dataset and convert sentiment labels
    into numerical labels.

    Positive -> 1
    Negative -> 0
    """

    df = pd.read_csv(path)

    required = {"review", "sentiment"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {required}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only required columns and remove missing values
    df = df[["review", "sentiment"]].dropna().copy()

    # Convert sentiment labels into numerical values
    mapping = {
        "positive": 1,
        "negative": 0,
        "pos": 1,
        "neg": 0,
    }

    df["label"] = (
        df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if df["label"].isna().any():
        raise ValueError(
            "Unexpected sentiment values. "
            "Use Positive/Negative labels."
        )

    df["label"] = df["label"].astype(int)

    return df


def balanced_sample(
    df: pd.DataFrame,
    sample_size: int | None
) -> pd.DataFrame:
    """
    Create a balanced subset for faster development/testing.

    Example:
    --sample 10000
    gives approximately:
        5000 positive
        5000 negative
    """

    # If sample is not provided or is larger than dataset,
    # use the complete dataset.
    if not sample_size or sample_size >= len(df):
        return df

    n_each = sample_size // 2

    parts = []

    for label in [0, 1]:

        part = df[df["label"] == label].sample(
            n=min(
                n_each,
                (df["label"] == label).sum()
            ),
            random_state=42,
        )

        parts.append(part)

    # Combine positive and negative samples
    # and shuffle them.
    out = (
        pd.concat(parts)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return out


def make_vectorizer(kind: str):
    """
    Create the requested text vectorizer.

    1-3 means:
        Unigram + Bigram + Trigram
    """

    if kind == "binary":

        return CountVectorizer(
            binary=True,
            ngram_range=(1, 3),
            min_df=2
        )

    if kind == "count":

        return CountVectorizer(
            binary=False,
            ngram_range=(1, 3),
            min_df=2
        )

    if kind == "tfidf":

        return TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2
        )

    raise ValueError(
        f"Unknown vectorizer: {kind}"
    )


def main():

    # ---------------------------------------------------------
    # Command-line arguments
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to IMDB_Dataset.csv"
    )

    parser.add_argument(
        "--output",
        default="artifacts",
        help="Output directory"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Optional balanced subset size "
            "for a faster development run."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
    )

    parser.add_argument(
        "--vectorizers",
        nargs="+",
        choices=[
            "binary",
            "count",
            "tfidf"
        ],
        default=[
            "binary",
            "count",
            "tfidf"
        ],
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    out = Path(args.output)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("Loading dataset...")

    df = load_dataset(args.data)

    # Optional balanced sampling
    df = balanced_sample(
        df,
        args.sample
    )

    print(f"\nTotal rows used: {len(df)}")

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .rename({
            0: "negative",
            1: "positive"
        })
    )

    # ---------------------------------------------------------
    # Clean reviews
    # ---------------------------------------------------------

    print("\nCleaning reviews...")

    cleaned = clean_corpus(
        df["review"].tolist()
    )

    # ---------------------------------------------------------
    # TRAIN / TEST SPLIT
    #
    # 80% TRAINING
    # 20% TESTING
    # ---------------------------------------------------------

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned,
        df["label"].to_numpy(),

        # 20% of the data is reserved for testing.
        # Therefore 80% is automatically used for training.
        test_size=0.20,

        random_state=42,

        # Maintains the same positive/negative
        # class distribution in both sets.
        stratify=df["label"],
    )

    # ---------------------------------------------------------
    # Display split information
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print(
        f"Total dataset : {len(cleaned)} reviews"
    )

    print(
        f"Training data : {len(X_train_text)} reviews "
        f"({len(X_train_text) / len(cleaned) * 100:.0f}%)"
    )

    print(
        f"Testing data  : {len(X_test_text)} reviews "
        f"({len(X_test_text) / len(cleaned) * 100:.0f}%)"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Display class distribution
    # ---------------------------------------------------------

    print("\nTraining class distribution:")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
        .rename({
            0: "negative",
            1: "positive"
        })
    )

    print("\nTesting class distribution:")

    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
        .rename({
            0: "negative",
            1: "positive"
        })
    )

    # ---------------------------------------------------------
    # Store results
    # ---------------------------------------------------------

    results = []

    trained_models = {}

    # ---------------------------------------------------------
    # Train models with different vectorizers
    # ---------------------------------------------------------

    for vectorizer_kind in args.vectorizers:

        print(
            f"\n\n{'=' * 60}"
        )

        print(
            f"VECTORING METHOD: {vectorizer_kind.upper()}"
        )

        print(
            f"{'=' * 60}"
        )

        # -----------------------------------------------------
        # Create vectorizer
        # -----------------------------------------------------

        vectorizer = make_vectorizer(
            vectorizer_kind
        )

        # -----------------------------------------------------
        # Vectorization
        # -----------------------------------------------------

        t0 = time.time()

        X_train = vectorizer.fit_transform(
            X_train_text
        )

        X_test = vectorizer.transform(
            X_test_text
        )

        vectorization_time = (
            time.time() - t0
        )

        print(
            f"Feature matrix:"
        )

        print(
            f"Training: {X_train.shape}"
        )

        print(
            f"Testing : {X_test.shape}"
        )

        print(
            f"Vectorization time: "
            f"{vectorization_time:.1f}s"
        )

        # -----------------------------------------------------
        # Save vectorizer
        # -----------------------------------------------------

        vectorizer_path = (
            out /
            f"vectorizer_{vectorizer_kind}.joblib"
        )

        joblib.dump(
            vectorizer,
            vectorizer_path
        )

        # -----------------------------------------------------
        # Train each model
        # -----------------------------------------------------

        for model_name in args.models:

            print(
                f"\nTraining "
                f"{model_name} + "
                f"{vectorizer_kind}..."
            )

            # Build model
            model = build_model(
                model_name,
                vectorizer_kind
            )

            # -------------------------------------------------
            # Train model
            # -------------------------------------------------

            t0 = time.time()

            model.fit(
                X_train,
                y_train
            )

            train_seconds = (
                time.time() - t0
            )

            # -------------------------------------------------
            # Evaluate model
            # -------------------------------------------------

            label = (
                f"{model_name}_"
                f"{vectorizer_kind}_3gram"
            )

            metrics = evaluate_model(
                model,
                X_test,
                y_test,
                out,
                label
            )

            # -------------------------------------------------
            # Store metrics
            # -------------------------------------------------

            row = {
                "model": model_name,

                "vectorization":
                    vectorizer_kind,

                "ngram":
                    "1-3",

                "positive_precision":
                    metrics[
                        "positive_precision"
                    ],

                "negative_precision":
                    metrics[
                        "negative_precision"
                    ],

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "train_seconds":
                    train_seconds,

                "tp":
                    metrics["tp"],

                "tn":
                    metrics["tn"],

                "fp":
                    metrics["fp"],

                "fn":
                    metrics["fn"],
            }

            results.append(row)

            # -------------------------------------------------
            # Save trained model
            # -------------------------------------------------

            model_key = (
                f"{model_name}_"
                f"{vectorizer_kind}"
            )

            model_path = (
                out /
                f"model_{model_key}.joblib"
            )

            joblib.dump(
                model,
                model_path
            )

            trained_models[
                model_key
            ] = str(model_path)

            # -------------------------------------------------
            # Print metrics
            # -------------------------------------------------

            print(
                f"accuracy="
                f"{metrics['accuracy']:.4f}"
                f" | positive_precision="
                f"{metrics['positive_precision']:.4f}"
                f" | negative_precision="
                f"{metrics['negative_precision']:.4f}"
                f" | time="
                f"{train_seconds:.1f}s"
            )

    # ---------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if not results_df.empty:

        results_df = (
            results_df
            .sort_values(
                "accuracy",
                ascending=False
            )
        )

    # Save results.csv

    results_path = (
        out / "results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ---------------------------------------------------------
    # Save best model information
    # ---------------------------------------------------------

    if len(results_df):

        best = results_df.iloc[0]

        best_key = (
            f"{best['model']}_"
            f"{best['vectorization']}"
        )

        best_model_info = {

            "model":
                best["model"],

            "vectorization":
                best["vectorization"],

            "model_path":
                trained_models[
                    best_key
                ],

            "vectorizer_path":
                str(
                    out /
                    (
                        "vectorizer_"
                        f"{best['vectorization']}"
                        ".joblib"
                    )
                ),

            "accuracy":
                float(
                    best["accuracy"]
                ),
        }

        with open(
            out / "best_model.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                best_model_info,
                f,
                indent=2
            )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print(
        "\n\n" +
        "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    if not results_df.empty:

        print(
            results_df[
                [
                    "model",
                    "vectorization",
                    "positive_precision",
                    "negative_precision",
                    "accuracy",
                ]
            ].to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Final dataset information
    # ---------------------------------------------------------

    print(
        "\n\n" +
        "=" * 70
    )

    print(
        "PROJECT DATASET SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total reviews : {len(cleaned)}"
    )

    print(
        f"Training      : {len(X_train_text)} "
        f"({len(X_train_text) / len(cleaned) * 100:.0f}%)"
    )

    print(
        f"Testing       : {len(X_test_text)} "
        f"({len(X_test_text) / len(cleaned) * 100:.0f}%)"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved everything to:"
    )

    print(
        out.resolve()
    )


if __name__ == "__main__":
    main()