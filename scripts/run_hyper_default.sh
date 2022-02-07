#!/usr/bin/env bash

set -e
set -x
config_path=$1
suffix=$2
data_path="/workspace/bonito/bonito/data/dna_r9.4.1"
seed_arr=(42 1000)
# lr_arr=(5e-4 1e-3 1e-4 5e-5)
lr_arr=(1e-4 4e-4 8e-4)


for(( i=0;i<${#seed_arr[@]};i++)) do
    for(( j=0;j<${#lr_arr[@]};j++)) do
	model_path="bonito/data/pes_bidir/trans_res_${suffix}_s${seed_arr[i]}_lr${lr_arr[j]}";
	bonito train ${model_path}  -f --directory $data_path --config ${config_path} --lr ${lr_arr[j]} --seed ${seed_arr[i]} --epoch 30 --batch 64 --grad-accum-split 2;
        done
done
