"""Post-hoc probes for z_good and z_bad from a simple shared split SAE."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.probe.probe_split_sae_latents as shared_probe


_parse_shared_args = shared_probe.parse_args


def parse_args():
    args = _parse_shared_args()
    args.model_family = "simple_shared"
    return args


def main():
    shared_probe.parse_args = parse_args
    shared_probe.main()


if __name__ == "__main__":
    main()
