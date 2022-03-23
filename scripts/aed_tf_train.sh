set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/transformer'
mkdir ${dst_dir} 
dst_path="${dst_dir}/tf_e6_d3_4e-4"

conf='bonito/models/configs/aed_tf.toml'
seed=42
lr=1e-3
bs=64
ep=20
log_path='log_aed_tf.txt'
sp_num=2

rm -r ${dst_path}  
# CUDA_VISIBLE_DEVICES=0 nohup bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp  > ${log_path} & 
CUDA_VISIBLE_DEVICES=3 bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --no-amp --grad-accum-split ${sp_num} 


