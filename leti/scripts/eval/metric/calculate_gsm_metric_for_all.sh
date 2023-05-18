#!/bin/bash

set -x
for dir in $(find data/t5x-model/ -name gsm_original* -type d -exec find {} -name generations.json \; | xargs -I {} dirname {}); do
    python3 leti/scripts/eval/metric/calculate_gsm_metric.py $dir
    gsutil cp $dir/results.json $GS_BUCKET_PREFIX/$dir/results.json
done
