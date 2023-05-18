#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

MODEL_SIZE=2B
# Paths
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX; GCLOUD_PROJECT: $GCLOUD_PROJECT"
# - input
DATA_DIR=data/processed/finetune/mbpp
HF_MODEL_DIR=data/models/flax-pretrained/actor/codegen-$MODEL_SIZE-mono
# - output (need to contain init_checkpoint in t5x checkpointer format)
OUTPUT_DIR=data/t5x-model/mbpp-ft/actor/codegen-$MODEL_SIZE-mono

# Parameters
NUM_PARTITIONS=8 # size of model parallelism sub-mesh
# 384 steps per epoch, 2 replica -> 192 batch size per replica
BATCH_SIZE=384 # per data-parallel replica (num_replicas = num_tpus / num_partitions)
NUM_MIRCOBATCHES=96 # split batch into microbatches for gradient accumulation (8 per batch)
LR=1e-4
WEIGHT_DECAY=0.01
EPOCHS=30
LOGGING_STEPS=1
SAVE_STEPS=3

# Wandb
export WANDB_TASK=${MODEL_SIZE}-mbpp-ft-lr${LR}-wd${WEIGHT_DECAY}-bs${BATCH_SIZE}-ep${EPOCHS}
# export WANDB_MODE=disabled

# Check checkpoint to startwith
GS_SOURCE_CKPT_DIR=$GS_BUCKET_PREFIX/data/t5x-model/flax-pretrained/rw_conditioned/codegen-$MODEL_SIZE-mono/checkpoint_0
GS_TARGET_CKPT_DIR=$GS_BUCKET_PREFIX/$OUTPUT_DIR/checkpoint_0
# if the target directory does not exist, copy the checkpoint from the source directory
target_not_exists=$(gsutil -q stat $GS_TARGET_CKPT_DIR/* || echo 1)
if [[ $target_not_exists == 1 ]]; then
    echo "Target initial checkpoint directory $GS_TARGET_CKPT_DIR on Google Storage does not exist. Copying from source directory: $GS_SOURCE_CKPT_DIR"
    gsutil -m cp -r $GS_SOURCE_CKPT_DIR $GS_TARGET_CKPT_DIR
fi

function sync_from_gs() {
    local dir=$1
    if [ ! -d $dir ]; then
        echo "Syncing $dir from Google Storage to local disk"
        mkdir -p $dir
        gsutil -m rsync -r $GS_BUCKET_PREFIX/$dir $dir
    fi
}

# Sync data to local disk if not already there
sync_from_gs $DATA_DIR
sync_from_gs $HF_MODEL_DIR

# We use constant LR
python3 \
    leti/scripts/model/run_ft.py \
    --model_name_or_path $HF_MODEL_DIR \
    --gs_output_dir $GS_BUCKET_PREFIX/$OUTPUT_DIR \
    --per_replica_batch_size $BATCH_SIZE \
    --num_microbatches $NUM_MIRCOBATCHES \
    --num_partitions $NUM_PARTITIONS \
    --do_train \
    --train_file $DATA_DIR/train.json \
    --do_eval \
    --validation_file $DATA_DIR/val.json \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --learning_rate_end $LR \
    --weight_decay $WEIGHT_DECAY \
    --num_train_epochs $EPOCHS \
    --preprocessing_num_workers 1
