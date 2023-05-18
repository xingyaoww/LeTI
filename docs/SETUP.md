
Released code in this repo are tailored to work with Google Cloud TPU and Google Storage. Feel free to adapt the released code to your specific computing setup.

This is a setup guide if you want to reproduce the experiments on Google Cloud.

# Setup Google Cloud Compute and Storage

You can request access to free cloud TPU resources through the [TRC program](https://sites.research.google/trc/about/). You can set up your Google Storage bucket following their [official documentation](https://cloud.google.com/storage/docs/creating-buckets).

You will obtain a Google Cloud Project ID and a bucket path (e.g., `gs://your-gs-bucket`), which we will use in later training and evaluation.

You should also install `gcloud` and finish login in your local terminal following [this](https://cloud.google.com/sdk/gcloud).

# Start an instance with training / evaluation job

1. Update `tpu-launch/prep-tpu-vm.sh` with your git information and the file/ssh keys you want to setup to the server. 

2. You can change the cloned git repo in `tpu-launch/entry_script.sh` to a repo that hosts code you want to execute.

3. You can start a TPU VM by running the `tpu-launch/start-trc-tpu.sh` script:

For example, to start a v3-8 instance named `v3-test-vm`, you can do the following

```bash
./tpu-launch/start-trc-tpu.sh v3-8 v3-test-vm
```

Optionally, you can also pass in a script to execute once the VM setup (e.g., installing dependencies) is complete, for example:

```bash
./tpu-launch/start-trc-tpu.sh v3-8 v3-test-vm ./script/to/execute
```

You can use any script for training and evaluation in [docs/TRAIN.md](./TRAIN.md) or [docs/EVAL.md](./EVAL.md).

Alternatively, you may directly ssh into the started VM to manually perform experiments:

```bash
gcloud compute tpus tpu-vm ssh --zone <your-zone> root@v3-test-vm
tmux a -t tpu # to attach into a tmux window where the `tpu-launch/entry_script.sh` will be executed.
# then do anything you want
```

`<your-zone>` info can be found in `tpu-launch/start-trc-tpu.sh`. It will use `us-central1-f` for `v2-8` and `europe-west4-a` for `v3-8`.


**Workflow**

- Executing `tpu-launch/start-trc-tpu.sh` will starts the TPU VM.
- It will execute `tpu-launch/prep-tpu-vm.sh` when the VM is successfully created, which will setup the VM (e.g., git config, copy ssh keys, setup tmux config and window) and execute `tpu-launch/entry_script.sh` in the tmux window of the created VM.
- `tpu-launch/entry_script.sh` will clone the code and setup the environment to perform experiments.
