#!/usr/bin/env bash

set -e
set -x
prefix="bonito/models/configs"
config_arr=("dna_r9.4.1@v_hg_trans_1.toml" "dna_r9.4.1@v_trans_1.toml" "dna_r9.4.1@v_trans_d512.toml" "dna_r9.4.1@v_trans_d512_hg.toml")
suf_arr=("_d768_hg_sch" "_d768" "_d512" "_d512_hg_sch")

for(( i=0;i<${#config_arr[@]};i++)) do
   CUDA_VISIBLE_DEVICES=${i} nohup bash run_hyper_default.sh ${prefix}/${config_arr[i]} ${suf_arr[i]} > "log_batch_log_${i}.txt" &
done

