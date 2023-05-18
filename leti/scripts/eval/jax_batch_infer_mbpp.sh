#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX"

# ======== Test set ========
export N_SAMPLES=16;
export BATCH_SIZE=128;
export SPLIT=test

export TOPP=1;
export TEMPERATURE=0.1;

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
        ./leti/scripts/eval/jax_infer_mbpp.sh $MODEL_PATH $SPLIT $CHECKPOINT_DIR
    done
}

function infer_model() {
    local MODEL_PATH=$1 # e.g., flax-pretrained/rw_conditioned/codegen-350M-mono
    local CKPT_PATH=$2 # e.g., checkpoint_0
    local DO_CONDITIONED=$3 # e.g., 0 or 1
    local SPLIT_MODE=$4 # 0: test only, 1: few-shot, 2: instruct-following-eval
    local EXTRA_ARGS=${@:5} # e.g., --num_partitions 8

    echo "==================================================="
    echo "Inferring on model: $MODEL_PATH, checkpoint: $CKPT"

    if [[ $SPLIT_MODE == 1 ]]; then
        ALL_SPLITS="test test_1shot test_3shot" 
        #  test_5shot test_10shot"
        echo "Inferring on few-shot splits: $ALL_SPLITS"
    elif [[ $SPLIT_MODE == 2 ]]; then
        ALL_SPLITS="instr-following-eval/mbpp-avoid-index-error-fb instr-following-eval/mbpp-cause-index-error-fb instr-following-eval/mbpp-avoid-syntax-error instr-following-eval/mbpp-avoid-type-error instr-following-eval/mbpp-avoid-index-error instr-following-eval/mbpp-cause-syntax-error instr-following-eval/mbpp-cause-type-error instr-following-eval/mbpp-cause-index-error"
        echo "Inferring on instruct-following-eval splits: $ALL_SPLITS"
    else
        ALL_SPLITS="test"
    fi

    for SPLIT in $ALL_SPLITS; do
        echo "Inferring on split: $SPLIT"

        ./leti/scripts/eval/jax_infer_mbpp.sh \
            $MODEL_PATH $SPLIT $CKPT_PATH \
            $EXTRA_ARGS
        
        if [[ $DO_CONDITIONED == 1 ]]; then
            ./leti/scripts/eval/jax_infer_mbpp.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_good \
                $EXTRA_ARGS

            ./leti/scripts/eval/jax_infer_mbpp.sh \
                $MODEL_PATH $SPLIT $CKPT_PATH \
                --conditioned_on_bad \
                $EXTRA_ARGS
        fi
    done

}

# === Normal Evaluation ===
# Baseline
infer_model flax-pretrained/rw_conditioned/codegen-350M-mono checkpoint_0 0 0
infer_model flax-pretrained/rw_conditioned/codegen-350M-mono checkpoint_0 0 2 # instruct-following-eval

# infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_3 0 0
infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_6 0 0
# infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_9 0 0
infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_12 0 0
# infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_15 0 0
infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_18 0 0
# infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_21 0 0
infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_24 0 0
# infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_27 0 0
infer_model mbpp-ft/actor/codegen-350M-mono checkpoint_30 0 0

infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+50x3+lr1e-5 checkpoint_31416 1 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+50x3+lr1e-5 checkpoint_31416 1 2 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+50x3+lr1e-5 checkpoint_15708 1 1
# infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+50x3+lr1e-5 checkpoint_15708 1 2 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+50x3+lr1e-5 checkpoint_8976 1 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+50x3+lr1e-5 checkpoint_15708 1 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+50x3+lr1e-5 checkpoint_15708 1 2 # instruct-following-eval

infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5 checkpoint_31416 1 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5 checkpoint_31416 1 2 # instruct-following-eval

# infer_model \
#     mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+postprocess+50x3+lr1e-5 \
#     checkpoint_8976 1 0

# infer_model \
#     mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+mixpretrain+postprocess+langfb+50x3+lr1e-5 \
#     checkpoint_13464 1 0

infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+onlypretrain+100x3+lr1e-5 checkpoint_15708 1 1
infer_model mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+onlypretrain+100x3+lr1e-5 checkpoint_15708 1 2 # instruct-following-eval
# === 2B ===
# Baseline

infer_model flax-pretrained/rw_conditioned/codegen-2B-mono checkpoint_0 0 0 \
    --num_partitions 4
infer_model flax-pretrained/rw_conditioned/codegen-2B-mono checkpoint_0 0 2 \
    --num_partitions 4 # instruct-following-eval

# infer_model mbpp-ft/actor/codegen-2B-mono/checkpoint_3 0 0 --num_partitions 4
infer_model mbpp-ft/actor/codegen-2B-mono checkpoint_6 0 0 --num_partitions 4
# infer_model mbpp-ft/actor/codegen-2B-mono/checkpoint_9 0 0 --num_partitions 4
infer_model mbpp-ft/actor/codegen-2B-mono checkpoint_12 0 0 --num_partitions 4
# infer_model mbpp-ft/actor/codegen-2B-mono/checkpoint_15 0 0 --num_partitions 4
infer_model mbpp-ft/actor/codegen-2B-mono checkpoint_18 0 0 --num_partitions 4
# infer_model mbpp-ft/actor/codegen-2B-mono/checkpoint_21 0 0 --num_partitions 4
infer_model mbpp-ft/actor/codegen-2B-mono checkpoint_24 0 0 --num_partitions 4
# infer_model mbpp-ft/actor/codegen-2B-mono/checkpoint_27 0 0 --num_partitions 4
infer_model mbpp-ft/actor/codegen-2B-mono checkpoint_30 0 0 --num_partitions 4

# Best ckpt for 2B+mixpretrain+10x3
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_17952 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_17952 1 2 \
#     --num_partitions 4 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_20196 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_20196 1 2 \
#     --num_partitions 4 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_20196 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr1e-5 checkpoint_20196 1 2 \
#     --num_partitions 4 # instruct-following-eval

infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr5e-6 checkpoint_13464 1 1 \
    --num_partitions 4
infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+10x3+lr5e-6 checkpoint_13464 1 2 \
    --num_partitions 4

# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+postprocess+langfb+30x3+lr1e-5 \
#     checkpoint_6732 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+postprocess+langfb+30x3+lr1e-5 \
#     checkpoint_6732 1 2 \
#     --num_partitions 4

# Same ckpt for 2B+30x3 & 2B+coarse-only+30x3
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+30x3+lr1e-5 checkpoint_8976 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+30x3+lr1e-5 checkpoint_8976 1 2 \
#     --num_partitions 4 # instruct-following-eval
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+30x3+lr1e-5 checkpoint_10098 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+30x3+lr1e-5 checkpoint_10098 1 2 \
#     --num_partitions 4 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+30x3+lr1e-5 checkpoint_8976 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+30x3+lr1e-5 checkpoint_8976 1 2 \
#     --num_partitions 4 # instruct-following-eval
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+30x3+lr1e-5 checkpoint_10098 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+30x3+lr1e-5 checkpoint_10098 1 2 \
#     --num_partitions 4 # instruct-following-eval

# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+30x3+lr1e-5 checkpoint_13464 1 1 \
#     --num_partitions 4
# infer_model mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+30x3+lr1e-5 checkpoint_13464 1 2 \
#     --num_partitions 4 # instruct-following-eval


infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+30x3+lr5e-6 \
    checkpoint_13464 1 1 \
    --num_partitions 4

infer_model \
    mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+coarse-only+mixpretrain+30x3+lr5e-6 \
    checkpoint_13464 1 2 \
    --num_partitions 4

# infer_model \
#     mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+postprocess+langfb+30x3+lr5e-6 \
#     checkpoint_6732 $SPLIT 1 \
#     --num_partitions 8

# infer_model \
#     mbpp-ft/actor-rw-conditioned/codegen-2B-mono/2B+rw_conditioned+mixpretrain+postprocess+30x3+lr5e-6 \
#     checkpoint_4488 $SPLIT 1 \
#     --num_partitions 8
