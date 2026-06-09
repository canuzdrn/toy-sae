"""Train separate-encoder split SAE with canonical-color good reconstruction.

This reuses the canonical-good training loop from the shared split-SAE script
but swaps in the separate-encoder model and optimizer/head-update helpers. It
keeps the same canonical target, bad-residual loss, and good-branch
full-reconstruction gradient gate.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.train.train_separate_split_sae_adv_update as separate_adv
import scripts.train.train_split_sae_canonical_good as canonical_good


_parse_canonical_args = canonical_good.parse_args
_default_shared_checkpoint_dir = Path("checkpoints/split_sae")


def parse_args():
    args = _parse_canonical_args(description=__doc__)
    if args.checkpoint_dir == _default_shared_checkpoint_dir:
        args.checkpoint_dir = Path("checkpoints/separate_split_sae")
    args.model_architecture = "separate_encoder"
    return args


def main():
    canonical_good.parse_args = parse_args
    canonical_good.make_model = separate_adv.make_model
    canonical_good.backbone_parameters = separate_adv.backbone_parameters
    canonical_good.color_head_parameters = separate_adv.color_head_parameters
    canonical_good.compute_losses = separate_adv.compute_losses
    canonical_good.latent_stats = separate_adv.latent_stats
    canonical_good.make_loader = separate_adv.make_loader
    canonical_good.set_backbone_trainable = separate_adv.set_backbone_trainable
    canonical_good.set_color_heads_trainable = separate_adv.set_color_heads_trainable
    canonical_good.train_adversary_heads = separate_adv.train_adversary_heads
    canonical_good.main()


if __name__ == "__main__":
    main()
