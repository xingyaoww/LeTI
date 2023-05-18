#!/bin/bash

source leti/scripts/model/train/rw_conditioned/mbpp/2B/source-jax-2B.sh

# Parameters
LR=5e-6 # constant learning rate
WEIGHT_DECAY=0.01
NUM_ITERATIONS=10
NUM_TRAIN_EPOCHS_PER_ITER=3

RM_TYPE="2B+rw_conditioned+onlypretrain"
DIR_NAME=$RM_TYPE+${NUM_ITERATIONS}"x"${NUM_TRAIN_EPOCHS_PER_ITER}+"lr"$LR

export EXP_ID=$DIR_NAME

train \
   data/t5x-model/mbpp-ft/actor-rw-conditioned/codegen-$MODEL_SIZE-mono/$DIR_NAME \
   --per_replica_batch_size 128 \
   --num_microbatches 16 \
   --num_partitions 8 \
   --learning_rate $LR \
   --learning_rate_end $LR \
   --weight_decay $WEIGHT_DECAY \
   --num_train_epochs $NUM_TRAIN_EPOCHS_PER_ITER \
   --num_iterations $NUM_ITERATIONS \
   --n_generations_per_input 128 \
   --logging_steps 10 \
   --save_intermediate \
   --only_pretrain_data \
   --pretrain_data_gs_path $GS_BUCKET_PREFIX/data/processed/pretrain/thestack-v1.1-50k
