#!/bin/bash

# Input args
MODEL_PATH=$1 # e.g., mbpp-ft/actor/codegen-350M-mono
# get e.g., actor/codegen-350M-mono as MODEL_NAME
MODEL_NAME=$(echo $MODEL_PATH | cut -d'/' -f 2,3)
SPLIT=$2 # e.g., test, train
CHECKPOINT_DIR=$3 # e.g., checkpoint_0
EXTRA_ARGS=${@:4}

# assert SPLITS in [test, train]
if [ "$SPLIT" != "test" ]; then
  echo "Inval split: $SPLIT"
  exit 1
fi
EVALUATE_TASK="gsm"

# Paths
echo "Using GS_BUCKET_PREFIX: $GS_BUCKET_PREFIX"
# - input
DATA_DIR=data/processed/eval/gsm
# use a pretrained model of the same size as the base
HF_MODEL_DIR=data/models/flax-pretrained/$MODEL_NAME
# use t5x checkpointer format to populate the actual parameters
GS_CKPT_DIR=$GS_BUCKET_PREFIX/data/t5x-model/$MODEL_PATH/$CHECKPOINT_DIR
GS_OUTPUT_DIR=$GS_CKPT_DIR-infer/$EVALUATE_TASK

INFER_DATA=$DATA_DIR/${SPLIT}_prompts.json
echo "** Infer data: $INFER_DATA"
echo "Evaluate task: $EVALUATE_TASK"
echo "HF model dir: $HF_MODEL_DIR"
echo "T5X Google storage checkpoint: $GS_CKPT_DIR"

MAX_LENGTH=1536
# These parameters are passed by env variables
# N_SAMPLES=16
# BATCH_SIZE=16
# TEMPERATURE=0.1
# TOPP=0.95

OUTPUT_DIR_SUFFIX=""
if [[ $EXTRA_ARGS == *"conditioned_on_good"* ]]; then
  OUTPUT_DIR_SUFFIX="-good"
elif [[ $EXTRA_ARGS == *"conditioned_on_bad"* ]]; then
  OUTPUT_DIR_SUFFIX="-bad"
fi

GS_OUTPUT_DIR=$GS_OUTPUT_DIR/n$N_SAMPLES-t$TEMPERATURE-topp${TOPP}${OUTPUT_DIR_SUFFIX}
echo "Output GS dir: $GS_OUTPUT_DIR"

function sync_from_gs() {
    local dir=$1
    if [ ! -d $dir ]; then
        echo "Syncing $dir from Google Storage to local disk"
        mkdir -p $dir
        gsutil -m rsync -r $GS_BUCKET_PREFIX/$dir $dir
    fi
}

# Inference on test prompts
# if generations.json not exists, run inference
target_not_exists=$(gsutil -q stat $GS_OUTPUT_DIR/pass_at_k.json || echo 1)
if [[ $target_not_exists == 1 ]]; then
  sync_from_gs $DATA_DIR
  sync_from_gs $HF_MODEL_DIR
  echo "Running inference since $GS_OUTPUT_DIR/pass_at_k.json does not exist"
  python3 leti/scripts/eval/jax_infer.py \
    --infer_data $INFER_DATA \
    --hf_ckpt $HF_MODEL_DIR \
    --t5x_path $GS_CKPT_DIR \
    --gs_output_dir $GS_OUTPUT_DIR \
    --max_len $MAX_LENGTH \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --top_p $TOPP \
    --pad_to_multiple_of 128 \
    --postprocess_type "gsm_pal" \
    $EXTRA_ARGS
fi

echo "Done"
