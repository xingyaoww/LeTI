# Porting PyTorch Checkpoint into T5X-style Jax Checkpoint

## 350M

```bash
# Build initial actor model ckpt
JAX_PLATFORMS="cpu" PYTHONPATH=`pwd`:$PYTHONPATH \
python3 leti/scripts/model/init/init_hf_codegen_flax.py \
    --model Salesforce/codegen-350M-mono \
    --output-dir data/models/flax-pretrained/rw_conditioned \
    --gs-output-dir $GS_BUCKET_PREFIX/data/t5x-model/flax-pretrained/rw_conditioned/codegen-350M-mono \
    --model-type rw_conditioned
```

## 2B

```bash
# Build initial actor model ckpt
JAX_PLATFORMS="cpu" PYTHONPATH=`pwd`:$PYTHONPATH \
python3 leti/scripts/model/init/init_hf_codegen_flax.py \
    --model Salesforce/codegen-2B-mono \
    --output-dir data/models/flax-pretrained/rw_conditioned \
    --gs-output-dir $GS_BUCKET_PREFIX/data/t5x-model/flax-pretrained/rw_conditioned/codegen-2B-mono \
    --model-type rw_conditioned
```
