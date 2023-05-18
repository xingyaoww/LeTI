# Dataset Setup

## Finetuning Dataset

### MBPP
```bash
python3 leti/scripts/data/process_mbpp.py \
    --output-dir data/processed/finetune/mbpp
```

## Evaluation Dataset

### HumanEval

```bash
# Note that human-eval is only for evaluation, but we still put it in the same folder.
python3 leti/scripts/data/process_humaneval.py \
    --output-dir data/processed/eval/humaneval
```

### GSM8K (PaL prompt)
`data/processed/raw/gsm8k-pal/gsm.jsonl` can be downloaded from [https://github.com/reasoning-machines/pal/blob/main/datasets/gsm.jsonl](https://github.com/reasoning-machines/pal/blob/main/datasets/gsm.jsonl).

```bash
python3 leti/scripts/data/process_gsm8k_pal.py \
    --input-file data/processed/raw/gsm8k-pal/gsm.jsonl \
    --output-dir data/processed/eval/gsm
```

### Big-Bench-Hard
```bash
mkdir external
git clone https://github.com/google/BIG-bench.git
cd ..

python3 leti/scripts/data/process_bbh.py \
    --input-repo-dir external/BIG-Bench-Hard \
    --output-dir data/processed/eval/bbh

python3 leti/scripts/data/process_bbh.py \
    --input-repo-dir external/BIG-Bench-Hard \
    --output-dir data/processed/eval/bbh_cot \
    --cot
```

### Event Argument Extraction

Following this [repo](https://github.com/xingyaoww/code4struct) to obtain `eae-v6.4-train.jsonl` and `eae-v6.4-test.jsonl`.
We are not able to provide it directly due to copyright restriction of ACE-05.

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python3 leti/scripts/data/process_eae.py \
    --train-file data/processed/raw/eae-v6.4-train.jsonl \
    --test-file data/processed/raw/eae-v6.4-test.jsonl \
    --output-dir data/processed/finetune/eae
```

## Pre-training Dataset

```bash
python3 leti/scripts/data/process_the_stack.py \
    --gs_project $GCLOUD_PROJECT \
    --output-dir data/processed/pretrain/thestack-v1.1-50k

gsutil cp -r data/processed/pretrain/thestack-v1.1-50k $GS_BUCKET_PREFIX/data/processed/pretrain/thestack-v1.1-50k
```
