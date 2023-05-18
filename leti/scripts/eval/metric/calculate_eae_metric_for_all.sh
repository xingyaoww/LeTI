#!/bin/bash

set -x
export PYTHONPATH=$PYTHONPATH:$(pwd)
for dir in $(find data/t5x-model/ -name eae* -type d -exec find {} -name generations.json \; | xargs -I {} dirname {}); do
    if [ -f $dir/results.json ]; then
        continue
    fi
    python3 leti/scripts/eval/metric/calculate_eae_metric.py $dir
    gsutil cp $dir/results.json $GS_BUCKET_PREFIX/$dir/results.json
done
