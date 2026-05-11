#!/usr/bin/env python
"""Train simple probes for digit, digit-group, and color information in embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EVAL_SPLITS = [
    "ae_val_balanced",
    "split_val_biased",
    "test_id_biased",
    "test_balanced",
    "test_reversed",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/base_ae_probes"))
    parser.add_argument("--train-split", default="ae_train_balanced")
    parser.add_argument("--eval-splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_embedding_split(embedding_dir, split_name):
    path = embedding_dir / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")

    data = np.load(path)
    split = {
        "embeddings": data["embeddings"].astype(np.float32),
        "digits": data["digits"].astype(np.int64),
        "colors": data["colors"].astype(np.int64),
        "digit_groups": data["digit_groups"].astype(np.int64),
        "metadata": str(data["metadata"]),
    }
    data.close()
    return split


def make_probe(max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def evaluate_probe(model, split, label_name):
    labels = split[label_name]
    predictions = model.predict(split["embeddings"])
    return float(accuracy_score(labels, predictions))


def main():
    args = parse_args()
    train = load_embedding_split(args.embedding_dir, args.train_split)
    print(f"Train split: {args.train_split}")
    print(f"Train embeddings: {train['embeddings'].shape}")

    digit_probe = make_probe(args.max_iter, args.seed)
    digit_group_probe = make_probe(args.max_iter, args.seed)
    color_probe = make_probe(args.max_iter, args.seed)

    print("Training digit probe: h -> digit")
    digit_probe.fit(train["embeddings"], train["digits"])
    print("Training digit-group probe: h -> digits 0-4 vs 5-9")
    digit_group_probe.fit(train["embeddings"], train["digit_groups"])
    print("Training color probe: h -> color")
    color_probe.fit(train["embeddings"], train["colors"])

    results = {
        "train_split": args.train_split,
        "embedding_dir": str(args.embedding_dir),
        "digit_probe": {},
        "digit_group_probe": {},
        "color_probe": {},
    }

    all_eval_splits = [args.train_split]
    for split_name in args.eval_splits:
        if split_name not in all_eval_splits:
            all_eval_splits.append(split_name)

    print("")
    print("Probe accuracy")
    print("split | digit_acc | group_acc | color_acc")
    print("--- | ---: | ---: | ---:")
    for split_name in all_eval_splits:
        split = train if split_name == args.train_split else load_embedding_split(args.embedding_dir, split_name)
        digit_acc = evaluate_probe(digit_probe, split, "digits")
        digit_group_acc = evaluate_probe(digit_group_probe, split, "digit_groups")
        color_acc = evaluate_probe(color_probe, split, "colors")
        results["digit_probe"][split_name] = digit_acc
        results["digit_group_probe"][split_name] = digit_group_acc
        results["color_probe"][split_name] = color_acc
        print(f"{split_name} | {digit_acc:.4f} | {digit_group_acc:.4f} | {color_acc:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "probe_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print("")
    print(f"Saved probe results to {results_path}")


if __name__ == "__main__":
    main()
