# Training

## Setup

You should setup the following environment variable before executing the training script.

```bash
export GCLOUD_PROJECT=your-gcloud-project
export GS_BUCKET_PREFIX=gs://your-gs-bucket/your-directory
```

## LeTI Training

Generally, each iteration (3 epochs) of LeTI takes about 22 hours for a 2B model, and 4 hours for 350M model on a TPU v3-8 VM instance.

### Experiment on MBPP

Execute the corresponding script to train the model:

- LeTI (350M): `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain.sh`
- LeTI (350M) w/o textual feedback: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+coarse-only+50x3+lr1e-5+mixpretrain.sh`

- LeTI (2B): `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain.sh`
- LeTI (2B) w/o textual feedback: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+coarse-only+30x3+lr5e-6+mixpretrain.sh`

### Ablations on MBPP

#### Changing the number of training problems |P|

You can compare the following results to baseline that uses full dataset (i.e., LeTI 350M or 2B). 

**2B**
- |P| = 16: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain+d16.sh`
- |P| = 64: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain+d64.sh`
- |P| = 128: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain+d128.sh`

**350M**
- |P| = 16: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain+d16.sh`
- |P| = 64: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain+d64.sh`
- |P| = 128: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain+d128.sh`

#### Changing the number of sampled solution `n` per problem 

You can compare the following results to baseline that uses n=128 (i.e., LeTI 350M or 2B). 

**2B**
- n=16: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain+n16.sh`
- n=64: `leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+mixpretrain+n64.sh`

**350M**
- n=16: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain+n16.sh`
- n=64: `leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+mixpretrain+n64.sh`

#### Remove Continued Pre-training Regularization

**2B**
`leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6.sh`

**350M**
`leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5.sh`

#### Only do Continued Pre-training

**2B**
`leti/scripts/model/train/rw_conditioned/mbpp/2B/finetune+10x3+lr5e-6+onlypretrain.sh`

**350M**
`leti/scripts/model/train/rw_conditioned/mbpp/350M/finetune+50x3+lr1e-5+onlypretrain.sh`

### Experiment on Event Argument Extraction (EAE)

You need to start an API server before start the training script using `leti/verifier/eae/run_api_server.sh`.

`leti/scripts/model/train/rw_conditioned/eae/350M/finetune+30x3+lr1e-5+n64.sh`

## Finetuned baselines for comparison

### MBPP Finetuning

**2B** `leti/scripts/model/train/actor/mbpp/jax-2B.sh`

**350M** `leti/scripts/model/train/actor/mbpp/jax-350M.sh`

