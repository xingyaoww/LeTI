# Evaluate Trained Model

You should setup the following environment variable before executing the evaluation script.

```bash
export GCLOUD_PROJECT=your-gcloud-project
export GS_BUCKET_PREFIX=gs://your-gs-bucket/your-directory
```

## Big-Bench-Hard

`leti/scripts/eval/jax_batch_infer_bbh.sh`

## GSM (PaL-prompt)

`leti/scripts/eval/jax_batch_infer_gsm.sh`

## HumanEval

`leti/scripts/eval/jax_batch_infer_humaneval.sh`

## Event Argument Extraction

This only evaluate model trained on EAE.

`leti/scripts/eval/jax_batch_infer_eae.sh`

# Where to find the evaluation results?

You may find the evaluation result in the corresponding folder on Google Storage.

For example, if you evaluate trained model `350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5` with `checkpoint_31416` on `gsm`, you can find the results in this directory:

`$GS_BUCKET_PREFIX/data/t5x-model/mbpp-ft/actor-rw-conditioned/codegen-350M-mono/350M+rw_conditioned+coarse-only+mixpretrain+50x3+lr1e-5/checkpoint_31416-infer/gsm/n40-t0.7-topp0.95`

You can find its result conditioned on good/bad reward tokens in `.../n40-t0.7-topp0.95-good` or `.../n40-t0.7-topp0.95-bad`.

The folder will contains a `pass_at_k.json` or `result.json` that contains the result.

