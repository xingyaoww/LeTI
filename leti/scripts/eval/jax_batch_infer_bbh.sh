#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX"

# ======== Test set ========
export BATCH_SIZE=64;

# Following PaL https://arxiv.org/pdf/2211.10435.pdf
export N_SAMPLES=1;
export TEMPERATURE=0.0;
export TOPP=1;

function infer_all_ckpt() {
    local MODEL_PATH=$1
    GS_CKPT_DIR=$GS_BUCKET_PREFIX/data/t5x-model/$MODEL_PATH
    echo "Inferring on all checkpoints under GS_CKPT_DIR: $GS_CKPT_DIR"

    # list all subdirectories that starts with "checkpoint_"
    # and does not ending or contain "-infer" or ".tmp"
    # e.g., checkpoint_0, checkpoint_1, checkpoint_2, ...
    gsutil ls $GS_CKPT_DIR | grep -E "checkpoint_[0-9]+\/" | sort -V | while read -r line; do
        CHECKPOINT_DIR=$(basename $line)
        echo "Processing CHECKPOINT_DIR: $CHECKPOINT_DIR"
        ./leti/scripts/eval/jax_infer_bbh.sh $MODEL_PATH $SPLIT $CHECKPOINT_DIR
    done
}

function infer_model() {
    local MODEL_PATH=$1 # e.g., flax-pretrained/rw_conditioned/codegen-350M-mono
    local CKPT_PATH=$2 # e.g., checkpoint_0
    local DO_CONDITIONED=$3 # e.g., 0 or 1
    local EXTRA_ARGS=${@:4} # e.g., --num_partitions 8

    echo "==================================================="
    echo "Inferring on model: $MODEL_PATH, checkpoint: $CKPT"

    ALL_SPLITS="test test-cot"

    for SPLIT in $ALL_SPLITS; do
        echo "Inferring on split: $SPLIT"

        ./leti/scripts/eval/jax_infer_bbh.sh \
            $MODEL_PATH $SPLIT $CKPT_PATH \
            $EXTRA_ARGS
        
        if [[ $DO_CONDITIONED == 1 ]]; then
            ./leti/scripts/eval/jax_infer_bbh.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_good \
                $EXTRA_ARGS

            ./leti/scripts/eval/jax_infer_bbh.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_bad \
                $EXTRA_ARGS
        fi
    done
}

infer_model flax-pretrained/rw_conditioned/codegen-350M-mono checkpoint_0 0
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+50x3+lr1e-5 \
    checkpoint_31416 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5 \
    checkpoint_31416 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+50x3+lr1e-5 \
    checkpoint_15708 1

# === 2B ===
export BATCH_SIZE=32;
infer_model flax-pretrained/rw_conditioned/codegen-2B-mono checkpoint_0 0 --num_partitions 8
infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr5e-6 checkpoint_13464 1 --num_partitions 8
infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+10x3+lr5e-6 checkpoint_13464 1 --num_partitions 8
infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+10x3+lr5e-6 checkpoint_6732 1 --num_partitions 8
