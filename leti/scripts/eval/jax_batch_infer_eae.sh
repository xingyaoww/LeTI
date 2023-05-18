#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX"

# ======== Test set ========
export SPLIT=test
export BATCH_SIZE=128;

# Following PaL https://arxiv.org/pdf/2211.10435.pdf
export N_SAMPLES=1;
export TOPP=1;
export TEMPERATURE=0.0;

function infer_model() {
    local MODEL_PATH=$1 # e.g., flax-pretrained/rw_conditioned/codegen-350M-mono
    local CKPT_PATH=$2 # e.g., checkpoint_0
    local SPLIT=$3 # e.g., test
    local DO_CONDITIONED=$4 # e.g., 0 or 1
    local EXTRA_ARGS=${@:5} # e.g., --num_partitions 8

    echo "==================================================="
    echo "Inferring on model: $MODEL_PATH, checkpoint: $CKPT"

    ./leti/scripts/eval/jax_infer_eae.sh \
        $MODEL_PATH $SPLIT $CKPT_PATH \
        $EXTRA_ARGS
    
    if [[ $DO_CONDITIONED == 1 ]]; then
        ./leti/scripts/eval/jax_infer_eae.sh \
            $MODEL_PATH $SPLIT $CKPT_PATH \
            --conditioned_on_good \
            $EXTRA_ARGS

        ./leti/scripts/eval/jax_infer_eae.sh \
            $MODEL_PATH $SPLIT $CKPT_PATH \
            --conditioned_on_bad \
            $EXTRA_ARGS
    fi

}

function infer_all_ckpt() {
    local MODEL_PATH=$1
    local SPLITS=$2
    local DO_CONDITIONED=$3
    local EXTRA_ARGS=${@:4}
    GS_CKPT_DIR=$GS_BUCKET_PREFIX/data/t5x-model/$MODEL_PATH
    echo "Inferring on all checkpoints under GS_CKPT_DIR: $GS_CKPT_DIR"

    # list all subdirectories that starts with "checkpoint_"
    # and does not ending or contain "-infer" or ".tmp"
    # e.g., checkpoint_0, checkpoint_1, checkpoint_2, ...
    gsutil ls $GS_CKPT_DIR | grep -E "checkpoint_[0-9]+\/" | sort -V | while read -r line; do
        CHECKPOINT_DIR=$(basename $line)
        echo "Processing CHECKPOINT_DIR: $CHECKPOINT_DIR"
        infer_model $MODEL_PATH $CHECKPOINT_DIR $SPLIT $DO_CONDITIONED \
            $EXTRA_ARGS
    done
}

# ===== 350M =====
infer_all_ckpt eae-ft/actor-rw-conditioned/codegen-350M-mono/eae+350M+rw_conditioned+30x3+lr1e-5+n64 $SPLIT 1
