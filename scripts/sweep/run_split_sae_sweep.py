"""Train shared-trunk split-SAE sweep configs one after another."""

import argparse
import json
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
    parser.add_argument("--config", type=Path, default=Path("configs/split_sae_sweep.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/split_sae"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_configs(path):
    path = project_path(path, PROJECT_ROOT)
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep config: {path}")

    configs = json.loads(path.read_text())
    if not isinstance(configs, list):
        raise ValueError("Sweep config must be a JSON list.")

    names = set()
    for config in configs:
        if "name" not in config:
            raise ValueError("Every sweep config needs a name.")
        if "train_script" not in config:
            raise ValueError(f"Config {config['name']} needs a train_script.")
        if config["name"] in names:
            raise ValueError(f"Duplicate sweep config name: {config['name']}")
        names.add(config["name"])

    return configs


def cli_name(key):
    return "--" + key.replace("_", "-")


def add_config_args(command, config):
    for key, value in config.items():
        if key in ["name", "train_script"]:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(cli_name(key))
            continue
        command.extend([cli_name(key), str(value)])


def build_train_command(config, checkpoint_dir, epochs, batch_size):
    train_script = project_path(config["train_script"], PROJECT_ROOT)
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training script: {train_script}")

    command = [
        sys.executable,
        str(train_script),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--deterministic",
    ]
    add_config_args(command, config)
    return command


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


def run_training(config, checkpoint_root, epochs, batch_size, overwrite):
    name = config["name"]
    checkpoint_dir = checkpoint_root / name
    history_path = checkpoint_dir / "history.json"
    log_path = checkpoint_dir / "train.log"
    command = build_train_command(config, checkpoint_dir, epochs, batch_size)

    save_json(
        checkpoint_dir / "sweep_config.json",
        {
            "config": config,
            "train_command": command,
        },
    )

    if history_path.exists() and not overwrite:
        print(f"Skipping {name}; {history_path} already exists.")
        return {
            "name": name,
            "status": "skipped_existing",
            "checkpoint_dir": str(checkpoint_dir),
            "log_path": str(log_path),
            "command": command,
        }

    return_code = run_command(command, log_path)
    return {
        "name": name,
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        "command": command,
    }


def main():
    args = parse_args()
    configs = load_configs(args.config)
    checkpoint_root = project_path(args.checkpoint_root, PROJECT_ROOT)
    summary = []
    start_time = time.time()

    for config in configs:
        print("")
        print("=" * 80)
        print(f"Training sweep config: {config['name']}")
        print("=" * 80)

        result = run_training(
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

    summary_path = checkpoint_root / "sweep_summary.json"
    save_json(
        summary_path,
        {
            "config_path": str(project_path(args.config, PROJECT_ROOT)),
            "elapsed_seconds": time.time() - start_time,
            "runs": summary,
        },
    )
    print("")
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
