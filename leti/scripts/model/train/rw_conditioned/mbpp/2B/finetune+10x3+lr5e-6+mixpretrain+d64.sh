#!/bin/bash

source leti/scripts/model/train/rw_conditioned/mbpp/2B/source-jax-2B.sh

# Parameters
LR=5e-6 # constant learning rate
WEIGHT_DECAY=0.01
NUM_ITERATIONS=10
NUM_TRAIN_EPOCHS_PER_ITER=3
NUM_TRAIN_SAMPLES=64

RM_TYPE="2B+rw_conditioned+mixpretrain"
DIR_NAME=$RM_TYPE+${NUM_ITERATIONS}"x"${NUM_TRAIN_EPOCHS_PER_ITER}+"lr"$LR+"d"$NUM_TRAIN_SAMPLES

export EXP_ID=$DIR_NAME

train \
   data/t5x-model/mbpp-ft/actor-rw-conditioned/codegen-$MODEL_SIZE-mono/$DIR_NAME \
   --per_replica_batch_size 128 \
   --num_microbatches 32 \
   --num_partitions 8 \
   --learning_rate $LR \
   --learning_rate_end $LR \
   --weight_decay $WEIGHT_DECAY \
   --num_train_epochs $NUM_TRAIN_EPOCHS_PER_ITER \
   --num_iterations $NUM_ITERATIONS \
   --n_generations_per_input 128 \
   --logging_steps 10 \
   --max_train_samples $NUM_TRAIN_SAMPLES \
   --save_intermediate \
   --mix_pretrain_data \
   --pretrain_data_gs_path $GS_BUCKET_PREFIX/data/processed/pretrain/thestack-v1.1-50k
