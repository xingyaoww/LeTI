#!/bin/bash

REGION=$1 # e.g., europe-west4-a
INSTANCE_NAME=$2 # e.g., xingyao6-trc-tpu-v3
CMDS=${@:3} # e.g., python3 xxx.py
LOGIN_NAME=root@$INSTANCE_NAME
echo "Logging to Instance $LOGIN_NAME"

ZONE="--zone $REGION"

# check if $EXTRA_SSH_ARGS is empty, if not, echo it
if [ -n "$EXTRA_SSH_ARGS" ]; then
  echo "Extra SSH Args: $EXTRA_SSH_ARGS"
fi

set -x

# ==== TODO: revise the following commands to suit your needs! ====
# Set git
gcloud compute tpus tpu-vm ssh $ZONE $EXTRA_SSH_ARGS $LOGIN_NAME --command 'git config --global user.name "<YOUR_NAME>"; git config --global user.email "<YOUR_EMAIL>"'
# for git clone via ssh
gcloud compute tpus tpu-vm scp $ZONE $EXTRA_SSH_ARGS ~/.ssh/gcp_github $LOGIN_NAME:~/.ssh/id_rsa
# for wandb config
gcloud compute tpus tpu-vm scp $ZONE $EXTRA_SSH_ARGS ~/.netrc $LOGIN_NAME:~ 
# Set tmux
gcloud compute tpus tpu-vm scp $ZONE $EXTRA_SSH_ARGS tpu-launch/files/.tmux.conf $LOGIN_NAME:~/
gcloud compute tpus tpu-vm scp $ZONE $EXTRA_SSH_ARGS tpu-launch/files/known_hosts $LOGIN_NAME:~/.ssh/
# ===================================================================

gcloud compute tpus tpu-vm scp $ZONE $EXTRA_SSH_ARGS tpu-launch/entry_script.sh $LOGIN_NAME:~/
# cd ~ and run ./entry_script.sh in tmux
gcloud compute tpus tpu-vm ssh $ZONE $EXTRA_SSH_ARGS $LOGIN_NAME --command "tmux new -s tpu -d; tmux send-keys -t tpu 'cd ~' C-m; tmux send-keys -t tpu './entry_script.sh $INSTANCE_NAME $CMDS' C-m"
