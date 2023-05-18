#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX"

# ======== Test set ========
# follow setting here:
# https://arxiv.org/pdf/2203.13474.pdf
# export N_SAMPLES=200;
# export BATCH_SIZE=128;
export N_SAMPLES=256;
export BATCH_SIZE=256;
export SPLIT=test
export TOPP=0.95;

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
        ./leti/scripts/eval/jax_infer_humaneval.sh $MODEL_PATH $SPLIT $CHECKPOINT_DIR
    done
}

function infer_model() {
    local MODEL_PATH=$1 # e.g., flax-pretrained/rw_conditioned/codegen-350M-mono
    local CKPT_PATH=$2 # e.g., checkpoint_0
    local SPLIT=$3 # e.g., test
    local DO_CONDITIONED=$4 # e.g., 0 or 1
    local EXTRA_ARGS=${@:5} # e.g., --num_partitions 8

    echo "==================================================="
    echo "Inferring on model: $MODEL_PATH, checkpoint: $CKPT"

    for TEMPERATURE in 0.2 0.6 0.8; do
        echo "Evaluating humaneval with temperature: $TEMPERATURE"
        export TEMPERATURE=$TEMPERATURE
        ./leti/scripts/eval/jax_infer_humaneval.sh \
            $MODEL_PATH $SPLIT $CKPT_PATH \
            $EXTRA_ARGS
        
        if [[ $DO_CONDITIONED == 1 ]]; then
            ./leti/scripts/eval/jax_infer_humaneval.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_good \
                $EXTRA_ARGS

            ./leti/scripts/eval/jax_infer_humaneval.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_bad \
                $EXTRA_ARGS
        fi
    done

}

# ===== 350M =====
# infer_model flax-pretrained/rw_conditioned/codegen-350M-mono checkpoint_0 $SPLIT 0
# pre-trained performance are reported from the original paper

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+50x3+lr1e-5 \
    checkpoint_31416 $SPLIT 1

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5 \
    checkpoint_31416 $SPLIT 1

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+50x3+lr1e-5 \
    checkpoint_15708 $SPLIT 1

# ===== 2B =====
export BATCH_SIZE=128;


infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr5e-6 \
    checkpoint_13464 $SPLIT 1 \
    --num_partitions 8

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+30x3+lr5e-6 \
    checkpoint_13464 $SPLIT 1 \
    --num_partitions 8

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+10x3+lr5e-6 \
    checkpoint_6732 $SPLIT 1 \
    --num_partitions 8
