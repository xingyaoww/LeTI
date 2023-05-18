#!/bin/bash

ACCELERATOR_TYPE=$1 # e.g., v2-8, v3-8
INSTANCE_NAME=$2
CMDS=${@:3} # e.g., python3 xxx.py

# check if "preemptible" is in $INSTANCE_NAME
if [[ $INSTANCE_NAME == *"preemptible"* ]]; then
  EXTRA_ARGS="--preemptible"
  echo "Creating preemptible VM"
else
  EXTRA_ARGS=""
fi

if [ $ACCELERATOR_TYPE == "v2-8" ]; then
  ZONE="us-central1-f"
  EXTRA_SSH_ARGS=""
elif [ $ACCELERATOR_TYPE == "v3-8" ]; then
  ZONE="europe-west4-a"
  EXTRA_SSH_ARGS=""
else
  echo "Please provide a valid accelerator type"
  exit 1
fi

# export EXTRA_SSH_ARGS as an environment variable
export EXTRA_SSH_ARGS;

# check if $INSTANCE_NAME is empty
if [ -z "$INSTANCE_NAME" ]; then
  echo "Please provide an instance name"
  exit 1
fi
echo "Creating VM $INSTANCE_NAME ($ACCELERATOR_TYPE) in zone $ZONE"

gcloud compute tpus tpu-vm create $INSTANCE_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=tpu-vm-tf-2.11.0 \
  $EXTRA_ARGS

# if success, prepare the VM
if [ $? -eq 0 ]; then
  echo "VM created successfully"
  ./tpu-launch/prep-tpu-vm.sh $ZONE $INSTANCE_NAME $CMDS
else
  echo "VM creation failed"
  exit 1
fi
