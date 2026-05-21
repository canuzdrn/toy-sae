#!/usr/bin/env python
"""Run post-hoc probes for completed separate split-SAE runs."""

import argparse
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.utils.script_utils import command_text, project_path, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/separate_split_sae"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/separate_split_sae_latent_probes"))
    parser.add_argument("--checkpoints", nargs="+", default=["best_recon"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_filename(name):
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


def checkpoint_label(name):
    return checkpoint_filename(name).removesuffix(".pt")


def completed_run_dirs(checkpoint_root):
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Missing checkpoint root: {checkpoint_root}")
    return sorted(path for path in checkpoint_root.iterdir() if path.is_dir())


def build_probe_command(checkpoint_path, out_dir):
    probe_script = PROJECT_ROOT / "scripts" / "probe" / "probe_separate_split_sae_latents.py"
    if not probe_script.exists():
        raise FileNotFoundError(f"Missing probe script: {probe_script}")

    return [
        sys.executable,
        str(probe_script),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
    ]


def existing_results_path(run_dir, out_root, label):
    candidates = [
        out_root / f"{run_dir.name}_{label}" / "probe_results.json",
    ]

    if label == "best_recon":
        candidates.append(out_root / run_dir.name / "probe_results.json")

    for path in candidates:
        if path.exists():
            return path
    return None


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = command_text(command)
    print(text)

    with log_path.open("w") as log_file:
        log_file.write(text + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return_code = process.wait()
        log_file.write(f"\nreturn_code={return_code}\n")
        return return_code


def run_probe(run_dir, checkpoint_name, out_root, overwrite):
    checkpoint_path = run_dir / checkpoint_filename(checkpoint_name)
    label = checkpoint_label(checkpoint_name)
    out_dir = out_root / f"{run_dir.name}_{label}"
    log_path = out_dir / "probe.log"

    if not checkpoint_path.exists():
        print(f"Skipping {run_dir.name}/{label}; missing {checkpoint_path.name}.")
        return {
            "run_name": run_dir.name,
            "checkpoint": str(checkpoint_path),
            "out_dir": str(out_dir),
            "status": "skipped_missing_checkpoint",
        }

    results_path = existing_results_path(run_dir, out_root, label)
    if results_path is not None and not overwrite:
        print(f"Skipping {run_dir.name}/{label}; {results_path} already exists.")
        return {
            "run_name": run_dir.name,
            "checkpoint": str(checkpoint_path),
            "out_dir": str(out_dir),
            "results_path": str(results_path),
            "status": "skipped_existing",
        }

    command = build_probe_command(checkpoint_path, out_dir)
    return_code = run_command(command, log_path)
    return {
        "run_name": run_dir.name,
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "command": command,
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
    }


def main():
    args = parse_args()
    checkpoint_root = project_path(args.checkpoint_root, PROJECT_ROOT)
    out_root = project_path(args.out_root, PROJECT_ROOT)
    summary = []
    start_time = time.time()

    for run_dir in completed_run_dirs(checkpoint_root):
        for checkpoint_name in args.checkpoints:
            label = checkpoint_label(checkpoint_name)
            print("")
            print("=" * 80)
            print(f"Probing run: {run_dir.name} ({label})")
            print("=" * 80)

            result = run_probe(
                run_dir=run_dir,
                checkpoint_name=checkpoint_name,
                out_root=out_root,
                overwrite=args.overwrite,
            )
            summary.append(result)

            if result["status"] == "failed":
                print(f"Stopping because probing failed for {run_dir.name}/{label}.")
                break

        if summary and summary[-1]["status"] == "failed":
            break

    summary_path = out_root / "probe_sweep_summary.json"
    save_json(
        summary_path,
        {
            "checkpoint_root": str(checkpoint_root),
            "out_root": str(out_root),
            "checkpoints": [checkpoint_filename(name) for name in args.checkpoints],
            "elapsed_seconds": time.time() - start_time,
            "runs": summary,
        },
    )
    print("")
    print(f"Saved probe sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
