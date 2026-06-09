"""Train canonical-good split-SAE sweep configs one after another.

This is a thin wrapper around the shared-trunk sweep runner with defaults for
the canonical-good config file.
"""

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.sweep.run_split_sae_sweep as split_sweep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/canonical_good_swep.json"),
        help="JSON list of canonical-good sweep runs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/split_sae"),
        help="Parent folder for sweep checkpoint directories.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = split_sweep.load_configs(args.config)
    checkpoint_root = split_sweep.project_path(args.checkpoint_root, split_sweep.PROJECT_ROOT)
    summary = []
    start_time = time.time()

    for config in configs:
        print("")
        print("=" * 80)
        print(f"Training canonical-good sweep config: {config['name']}")
        print("=" * 80)

        result = split_sweep.run_training(
            config=config,
            checkpoint_root=checkpoint_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        summary.append(result)

        if result["status"] == "failed":
            print(f"Stopping because {config['name']} failed.")
            break

    summary_path = checkpoint_root / "canonical_good_sweep_summary.json"
    split_sweep.save_json(
        summary_path,
        {
            "config_path": str(split_sweep.project_path(args.config, split_sweep.PROJECT_ROOT)),
            "elapsed_seconds": time.time() - start_time,
            "runs": summary,
        },
    )
    print("")
    print(f"Saved canonical-good sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
