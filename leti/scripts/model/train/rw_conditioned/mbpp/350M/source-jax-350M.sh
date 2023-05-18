#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=17179869184 # 16GB
ulimit -n 1000000 # increase the number of open files

MODEL_SIZE=350M
export WANDB_TASK=${MODEL_SIZE}-mbpp-rw-conditioned-ft

# export JAX_PLATFORMS=cpu # for debugging
# export WANDB_MODE=disabled
# constant path (input)
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX; GCLOUD_PROJECT: $GCLOUD_PROJECT"
DATA_DIR=data/processed/finetune/mbpp
ACTOR_HF_MODEL_DIR=data/models/flax-pretrained/rw_conditioned/codegen-$MODEL_SIZE-mono

function sync_from_gs() {
    local dir=$1
    if [ ! -d $dir ]; then
        echo "Syncing $dir from Google Storage to local disk"
        mkdir -p $dir
        gsutil -m rsync -r $GS_BUCKET_PREFIX/$dir $dir
    fi
}

function train() {
    # kill previous runs
    ps aux | grep run_clm_flax | awk '{print $2}' | xargs kill -9
    
    OUTPUT_DIR=$1
    EXTRA_ARGS=${@:2}

    # Check checkpoint to startwith
    GS_SOURCE_CKPT_DIR=$GS_BUCKET_PREFIX/data/t5x-model/flax-pretrained/rw_conditioned/codegen-$MODEL_SIZE-mono/checkpoint_0
    GS_TARGET_CKPT_DIR=$GS_BUCKET_PREFIX/$OUTPUT_DIR/checkpoint_0
    # if the target directory does not exist, copy the checkpoint from the source directory
    target_not_exists=$(gsutil -q stat $GS_TARGET_CKPT_DIR/* || echo 1)
    if [[ $target_not_exists == 1 ]]; then
        echo "Target initial checkpoint directory $GS_TARGET_CKPT_DIR on Google Storage does not exist. Copying from source directory: $GS_SOURCE_CKPT_DIR"
        gsutil -m cp -r $GS_SOURCE_CKPT_DIR $GS_TARGET_CKPT_DIR
    fi

    # Sync data to local disk if not already there
    sync_from_gs $DATA_DIR
    sync_from_gs $ACTOR_HF_MODEL_DIR

    # fetch WANDB_RUN_ID if it exists
    RUN_ID=$(./tpu-launch/get_wandb_latest_run.py $EXP_ID)
    if [[ $RUN_ID != "" ]]; then
        export WANDB_RUN_ID=$RUN_ID
        echo "Found existing run id: $RUN_ID"
    fi

    python3 \
        leti/scripts/model/run_leti.py \
        --gs_output_dir $GS_BUCKET_PREFIX/$OUTPUT_DIR \
        --actor_model_name_or_path $ACTOR_HF_MODEL_DIR \
        --do_train \
        --train_file $DATA_DIR/train_prompts.json \
        --do_eval \
        --validation_file $DATA_DIR/val.json \
        --do_test \
        --test_file $DATA_DIR/test_prompts.json \
        --preprocessing_num_workers 8 \
        $EXTRA_ARGS
}
