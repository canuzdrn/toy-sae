set -euo pipefail

TRAIN_SCRIPT="scripts/train/train_split_sae_adv_update.py"
PROBE_SCRIPT=""
VIZ_SCRIPT="scripts/debug/visualize_reconstructions.py"
CHECKPOINT_DIR=""
CHECKPOINT_NAME="best_recon"
PROBE_OUT_DIR=""
VIZ_SPLIT="test_balanced"
VIZ_INDICES=(6279 7898 1059 8375)
SKIP_TRAIN=0
SKIP_PROBE=0
SKIP_VIZ=0
TRAIN_ARGS=()

usage() {
    cat <<'EOF'
Run training, post-hoc probing, and reconstruction visualization sequentially.

Usage:
  bash scripts/run_train_probe_visualize.sh \
    --checkpoint-dir checkpoints/split_sae/my_run \
    [wrapper options] \
    -- [training arguments]

Wrapper options:
  --train-script PATH       Training script to run.
                            Default: scripts/train/train_split_sae_adv_update.py
  --checkpoint-dir DIR      Checkpoint folder for the run. Required.
  --checkpoint-name NAME    Checkpoint stem to probe/visualize.
                            Default: best_recon
  --probe-script PATH       Probe script. If omitted, inferred from checkpoint/train path.
  --probe-out-dir DIR       Probe output folder. If omitted, inferred from run name.
  --viz-script PATH         Reconstruction visualization script.
                            Default: scripts/debug/visualize_reconstructions.py
  --viz-split SPLIT         Split for reconstruction visualization.
                            Default: test_balanced
  --viz-indices I J K L     Fixed example indices for visualization.
                            Default: 6279 7898 1059 8375
  --skip-train              Do not run training.
  --skip-probe              Do not run post-hoc probing.
  --skip-viz                Do not run reconstruction visualization.
  --help                    Show this help.

Everything after "--" is passed directly to the training script.
Do not include --checkpoint-dir after "--"; this wrapper supplies it.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-script)
            TRAIN_SCRIPT="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --checkpoint-name)
            CHECKPOINT_NAME="$2"
            shift 2
            ;;
        --probe-script)
            PROBE_SCRIPT="$2"
            shift 2
            ;;
        --probe-out-dir)
            PROBE_OUT_DIR="$2"
            shift 2
            ;;
        --viz-script)
            VIZ_SCRIPT="$2"
            shift 2
            ;;
        --viz-split)
            VIZ_SPLIT="$2"
            shift 2
            ;;
        --viz-indices)
            shift
            VIZ_INDICES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                VIZ_INDICES+=("$1")
                shift
            done
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --skip-probe)
            SKIP_PROBE=1
            shift
            ;;
        --skip-viz)
            SKIP_VIZ=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown wrapper option: $1" >&2
            echo >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "--checkpoint-dir is required." >&2
    echo >&2
    usage >&2
    exit 2
fi

if [[ ${#VIZ_INDICES[@]} -eq 0 ]]; then
    echo "--viz-indices requires at least one index." >&2
    exit 2
fi

RUN_NAME="$(basename "$CHECKPOINT_DIR")"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_NAME.pt"

if [[ -z "$PROBE_SCRIPT" ]]; then
    if [[ "$CHECKPOINT_DIR" == *"separate_split_sae"* || "$TRAIN_SCRIPT" == *"separate_split_sae"* ]]; then
        PROBE_SCRIPT="scripts/probe/probe_separate_split_sae_latents.py"
    else
        PROBE_SCRIPT="scripts/probe/probe_split_sae_latents.py"
    fi
fi

if [[ -z "$PROBE_OUT_DIR" ]]; then
    if [[ "$PROBE_SCRIPT" == *"probe_separate_split_sae_latents.py" ]]; then
        PROBE_OUT_DIR="outputs/separate_split_sae_latent_probes/${RUN_NAME}_${CHECKPOINT_NAME}"
    else
        PROBE_OUT_DIR="outputs/split_sae_latent_probes/${RUN_NAME}_${CHECKPOINT_NAME}"
    fi
fi

echo "Run name: $RUN_NAME"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Checkpoint name: $CHECKPOINT_NAME"
echo "Training script: $TRAIN_SCRIPT"
echo "Probe script: $PROBE_SCRIPT"
echo "Probe output dir: $PROBE_OUT_DIR"
echo "Visualization split: $VIZ_SPLIT"
echo "Visualization indices: ${VIZ_INDICES[*]}"
echo

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo "=== Training ==="
    python "$TRAIN_SCRIPT" --checkpoint-dir "$CHECKPOINT_DIR" "${TRAIN_ARGS[@]}"
    echo
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Expected checkpoint does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ "$SKIP_PROBE" -eq 0 ]]; then
    echo "=== Post-hoc probing ==="
    python "$PROBE_SCRIPT" \
        --checkpoint "$CHECKPOINT_PATH" \
        --out-dir "$PROBE_OUT_DIR"
    echo
fi

if [[ "$SKIP_VIZ" -eq 0 ]]; then
    echo "=== Reconstruction visualization ==="
    python "$VIZ_SCRIPT" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --checkpoint-name "$CHECKPOINT_NAME" \
        --split "$VIZ_SPLIT" \
        --indices "${VIZ_INDICES[@]}"
fi
