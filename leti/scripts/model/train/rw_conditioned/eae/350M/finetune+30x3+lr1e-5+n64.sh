#!/bin/bash

source leti/scripts/model/train/rw_conditioned/eae/350M/source-jax-350M.sh

# Parameters
LR=1e-5 # constant learning rate
WEIGHT_DECAY=0.01
NUM_ITERATIONS=30
NUM_TRAIN_EPOCHS_PER_ITER=3
NUM_GENERATIONS_PER_INPUT=64

RM_TYPE="eae+350M+rw_conditioned"
DIR_NAME=$RM_TYPE+${NUM_ITERATIONS}"x"${NUM_TRAIN_EPOCHS_PER_ITER}+"lr"$LR+"n"$NUM_GENERATIONS_PER_INPUT

export EXP_ID=$DIR_NAME

train \
   data/t5x-model/eae-ft/actor-rw-conditioned/codegen-$MODEL_SIZE-mono/$DIR_NAME \
   --per_replica_batch_size 16 \
   --num_microbatches 16 \
   --num_partitions 1 \
   --learning_rate $LR \
   --learning_rate_end $LR \
   --weight_decay $WEIGHT_DECAY \
   --num_train_epochs $NUM_TRAIN_EPOCHS_PER_ITER \
   --num_iterations $NUM_ITERATIONS \
   --n_generations_per_input $NUM_GENERATIONS_PER_INPUT \
   --logging_steps 10 \
   --save_intermediate
