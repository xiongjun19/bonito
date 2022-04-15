set -x
data_path='/workspace/bonito/bonito/data/dna_r9.4.1'

dst_dir='bonito/data/transformer'
mkdir ${dst_dir} 
dst_path="${dst_dir}/tf_e6_d3_ft_v2"

conf='bonito/models/configs/aed_tf_joint.toml'
seed=42
lr=3e-4
bs=32
ep=30
log_path='log_aed_tf.txt'
sp_num=1

device=2 

rm -r ${dst_path}  
CUDA_VISIBLE_DEVICES=$device bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --grad-accum-split ${sp_num} 

# CUDA_VISIBLE_DEVICES=$device bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --grad-accum-split ${sp_num} --chunks 100

# conf='bonito/models/configs/aed_tf_joint.toml'
# seed=42
# lr=1e-4
# bs=32
# ep=20
# log_path='log_aed_tf.txt'
# sp_num=1
# 
# CUDA_VISIBLE_DEVICES=$device bonito train ${dst_path} -f --directory ${data_path} --config ${conf}  --lr $lr --epoch $ep --batch $bs --seed ${seed} --grad-accum-split ${sp_num} 

