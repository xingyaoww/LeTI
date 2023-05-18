#!/bin/bash

INSTANCE_NAME=$1
CMDS=${@:2}

echo "Intializing TPU VM: $INSTANCE_NAME"
# Initialize (clone, dependencies, etc.)
pushd ~
git clone https://github.com/xingyaoww/LeTI.git
pushd LeTI
pip3 install -U -r requirements-tpu.txt

sudo apt update
sudo apt install golang -y

# This is a hack to make sure torch_xla does not cause any issues
pip3 uninstall torch_xla -y
python3 -m spacy download en_core_web_sm

# do stuff here
# Run the command
echo "Running command: $CMDS"
$CMDS

# Stop current TPU VM
# export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google');
# export ZONE=$(echo $ZONE | cut -d/ -f4); # cut to the last chunk
# gcloud --quiet compute tpus tpu-vm delete $INSTANCE_NAME --zone=$ZONE
