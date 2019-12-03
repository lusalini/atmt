#!/bin/bash

max=10
beamsize=4
alpha=1

touch "results_k$beamsize.task4.txt"
mkdir -p "output_k$beamsize/translated"
mkdir -p "output_k$beamsize/postprocessed"

for gamma in {0..10}
do
echo "======  $gamma time  =====" >> results_k$beamsize.task4.txt
/usr/bin/python3.6 translate_beam.py --gamma-i $gamma --gamma-max $max --cuda 0 --beam-size=$beamsize --alpha-i=$alpha --alpha-max=$max --output=./output_k$beamsize/translated/model_translations_$alpha.txt
bash postprocess_asg4.sh ./output_k$beamsize/translated/model_translations_$alpha.txt ./output_k$beamsize/postprocessed/model_translations_$alpha.out en
cat ./output_k$beamsize/postprocessed/model_translations_$alpha.out | sacrebleu data_asg4/raw_data/test.en >> results_k$beamsize.task4.txt
done
